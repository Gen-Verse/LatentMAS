import os
import csv
import importlib.util
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


try:
    # Avoid accidentally importing a vendored `vllm/` inside the repo.
    spec = importlib.util.find_spec("vllm")
    if spec is None or (spec.origin and os.getcwd() in os.path.abspath(spec.origin)):
        raise ImportError("vllm not available or vendored locally; skipping vllm backend")
    from vllm import LLM
    # SamplingParams location can vary between vLLM versions; try common locations
    try:
        from vllm import SamplingParams
    except Exception:
        try:
            from vllm.sampling_params import SamplingParams
        except Exception:
            from dataclasses import dataclass

            @dataclass
            class SamplingParams:
                temperature: float = 0.7
                top_p: float = 0.95
                max_tokens: int = 256

    _HAS_VLLM = True
except Exception:
    LLM = None
    SamplingParams = None
    _HAS_VLLM = False


# Optional llama.cpp backend for fast (GGUF) inference. Imported lazily/optionally
# so the dependency is not required unless `--use_llamacpp` is requested.
try:
    from llama_cpp import Llama
    _HAS_LLAMACPP = True
except Exception:
    Llama = None
    _HAS_LLAMACPP = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"


def _default_model_dtype(device) -> torch.dtype:
    if device == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(device, "type") and device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    # Handle different past_key_values types:
    # - legacy tuple of tuples returned by transformers
    # - Cache/DynamicCache-like objects from vLLM or newer HF internals
    try:
        # If object provides a legacy conversion (e.g., Cache), convert and read
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            k = legacy[0][0]
            return k.shape[-2]
    except Exception:
        pass

    # If it's subscriptable (tuple/list of layers)
    try:
        if hasattr(past_key_values, "__getitem__"):
            k = past_key_values[0][0]
            return k.shape[-2]
    except Exception:
        pass

    # As a last resort, try to find the first tensor-like object and infer length
    try:
        # iterate if it's iterable
        for layer in past_key_values:
            if isinstance(layer, tuple) or isinstance(layer, list):
                for t in layer:
                    if hasattr(t, "shape"):
                        return t.shape[-2]
            elif hasattr(layer, "shape"):
                return layer.shape[-2]
    except Exception:
        pass

    return 0


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, use_llamacpp: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.use_llamacpp = use_llamacpp
        self.vllm_engine = None
        self.llamacpp_engine = None
        # HF_device: the device where the HF (transformers) model should be placed
        # Ensure the attribute always exists to avoid AttributeError in latent/vllm paths.
        self.HF_device = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.language_reasoning_disentangle = bool(getattr(args, "language_reasoning_disentangle", False)) if args else False
        self._lr_vector = None
        self._lr_vector_path = getattr(args, "lr_vector_path", None) if args else None
        self._lr_current_agent_role = None
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_llamacpp:
            if not _HAS_LLAMACPP:
                raise ImportError(
                    "llama-cpp-python is not installed. Install the optional extra with "
                    "`pip install -e .[llamacpp]` (or `pip install llama-cpp-python`) to use --use_llamacpp."
                )
            if self.latent_space_realign:
                raise ValueError(
                    "latent_space_realign / latent_mas are not supported by the llama.cpp backend "
                    "(GGUF inference does not expose hidden states). Use the transformers or vLLM backend."
                )
            model_path = getattr(args, "llamacpp_model_path", None)
            if not model_path:
                raise ValueError(
                    "--use_llamacpp requires --llamacpp_model_path pointing to a local GGUF file."
                )
            print(f"[llama.cpp] Loading GGUF model from {model_path}")
            self.llamacpp_engine = Llama(
                model_path=model_path,
                n_ctx=int(getattr(args, "llamacpp_n_ctx", 4096)),
                n_gpu_layers=int(getattr(args, "llamacpp_n_gpu_layers", -1)),
                n_threads=getattr(args, "llamacpp_n_threads", None),
                seed=int(getattr(args, "seed", 0)) if args else 0,
                verbose=bool(getattr(args, "llamacpp_verbose", False)),
            )
            # A HF tokenizer is still used to render chat templates consistently
            # with the rest of the pipeline; generation itself runs in llama.cpp.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            _ensure_pad_token(self.tokenizer)
            return  # skip loading the transformers model

        # --- ORCHESTRATION VIA COMPUTE SCAN ---
        import json
        scan_file = "compute_scan.json"
        num_gpus = 0
        total_mem_gb = 0
        if os.path.exists(scan_file):
            try:
                with open(scan_file, "r") as f:
                    scan = json.load(f)
                    num_gpus = scan.get("device_count", 0)
                    devices = scan.get("devices", [])
                    total_mem_gb = sum(d.get("total_memory_gb", 0) for d in devices)
                    
                    if num_gpus > 1 and str(device) != "auto" and not self.use_vllm:
                        print(f"[Orchestration] Detected {num_gpus} GPUs. Forcing device_map='auto' for accelerate.")
                        device = "auto"
            except Exception as e:
                print(f"[Orchestration] Failed to parse compute_scan.json: {e}")

        use_quantization = False
        if total_mem_gb > 0 and total_mem_gb < 24 and str(device) == "auto" and not self.use_vllm:
            print("[Orchestration] Low memory detected, enabling bitsandbytes 8-bit quantization.")
            use_quantization = True

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                dev2 = args.device2
                dtype2 = _default_model_dtype(dev2)
                if dev2 == "auto":
                    self.HF_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype2,
                        device_map="auto",
                    ).eval()
                    self.HF_device = self.HF_model.device
                else:
                    self.HF_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype2,
                    ).to(dev2).eval() 
                    self.HF_device = dev2
                self.embedding_layer = self.HF_model.get_input_embeddings()
                real_dev2 = torch.device(self.HF_device) if isinstance(self.HF_device, str) else self.HF_device
                self._ensure_latent_realign_matrix(self.HF_model, real_dev2, args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        
        # Optional Unsloth support
        if "unsloth" in model_name.lower():
            try:
                from unsloth import FastLanguageModel
                print("[Orchestration] Unsloth model detected. Using FastLanguageModel.")
                with torch.no_grad():
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=model_name,
                        dtype=_default_model_dtype(device),
                        load_in_8bit=use_quantization,
                        device_map="auto" if device == "auto" else {"": device}
                    )
            except ImportError as exc:
                raise ImportError(
                    "Unsloth model requested but 'unsloth' package is not installed. "
                    "Refusing to fall back to default transformers."
                ) from exc

        if not hasattr(self, "model"):
            with torch.no_grad():
                if device == "auto":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=_default_model_dtype(device),
                        device_map="auto",
                        load_in_8bit=use_quantization,
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=_default_model_dtype(device),
                        load_in_8bit=use_quantization,
                    )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        if device == "auto":
            self.device = self.model.device
        else:
            self.model.to(device)
            self.device = device
            
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def set_current_agent_role(self, role: Optional[str]) -> None:
        self._lr_current_agent_role = role

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations

    def llamacpp_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.llamacpp_engine:
            raise RuntimeError(
                "llama.cpp engine not initialized. Pass use_llamacpp=True to ModelWrapper."
            )
        # llama-cpp-python evaluates one prompt at a time; iterate over the batch.
        generations: List[str] = []
        for prompt in prompts:
            out = self.llamacpp_engine(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            generations.append(out["choices"][0]["text"].strip())
        return generations

    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    def _load_language_reasoning_vector(self, device: torch.device) -> torch.Tensor:
        if not self._lr_vector_path:
            raise ValueError("--language_reasoning_disentangle requires --lr_vector_path.")
        if self._lr_vector is None:
            vector = torch.load(self._lr_vector_path, map_location="cpu")
            if isinstance(vector, dict):
                vector = vector.get("vector")
            if not torch.is_tensor(vector):
                raise ValueError(f"Language-reasoning vector at {self._lr_vector_path} is not a tensor.")
            if vector.dim() != 3:
                raise ValueError(
                    "Language-reasoning vector must have shape [layers, rank, hidden_dim]. "
                    f"Got {tuple(vector.shape)}."
                )
            self._lr_vector = vector.detach().to(torch.float32)
        return self._lr_vector.to(device=device, dtype=torch.float32)

    def _language_reasoning_roles(self) -> set:
        roles = getattr(self.args, "lr_disentangle_roles", "planner,critic,refiner")
        return {x.strip().lower() for x in roles.split(",") if x.strip()}

    def _should_apply_language_reasoning_disentangle(self) -> bool:
        if not self.language_reasoning_disentangle:
            return False
        role = (self._lr_current_agent_role or "").lower()
        return role in self._language_reasoning_roles()

    def _apply_language_reasoning_disentangle(self, hidden: torch.Tensor) -> torch.Tensor:
        if not self._should_apply_language_reasoning_disentangle():
            return hidden

        vector = self._load_language_reasoning_vector(hidden.device)
        if vector.shape[-1] != hidden.shape[-1]:
            raise ValueError(
                "Language-reasoning vector hidden dimension does not match model hidden dimension. "
                f"Vector has {vector.shape[-1]}, hidden has {hidden.shape[-1]}. "
                "Build a vector for this exact model/traces before enabling disentanglement."
            )

        layer_idx = int(getattr(self.args, "lr_disentangle_vector_layer", -1))
        layer_idx = layer_idx if layer_idx >= 0 else vector.shape[0] + layer_idx
        if layer_idx < 0 or layer_idx >= vector.shape[0]:
            raise ValueError(f"--lr_disentangle_vector_layer {layer_idx} is out of range for vector with {vector.shape[0]} layers.")

        dirs = vector[layer_idx]
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-6)
        hidden_fp32 = hidden.to(torch.float32)
        projection = torch.matmul(torch.matmul(hidden_fp32, dirs.T), dirs)
        strength = float(getattr(self.args, "lr_disentangle_strength", 0.2))
        steered = hidden_fp32 - strength * projection
        return steered.to(hidden.dtype)

    def _prepare_latent_vec(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        latent_vec = self._apply_latent_realignment(hidden, model)
        return self._apply_language_reasoning_disentangle(latent_vec)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_length = input_ids.shape[-1]
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        # Transformers' `generate()` may not accept a `cache_position` kwarg
        # (it's used by vLLM). Only pass supported kwargs; we already
        # adjusted `attention_mask` when `past_key_values` is provided so
        # generation aligns correctly.
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        if past_key_values is not None:
            gen_kwargs["past_key_values"] = past_key_values

        outputs = self.model.generate(**gen_kwargs)
        sequences = outputs.sequences
        generations: List[str] = []
        for idx in range(sequences.shape[0]):
            generated_ids = sequences[idx, prompt_length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    def _last_token_hidden_by_layer(self, outputs, last_token_index: int) -> torch.Tensor:
        hidden_by_layer = []
        final_norm = getattr(getattr(self.model, "model", None), "norm", None)
        norm_device = None
        if final_norm is not None:
            try:
                norm_device = next(final_norm.parameters()).device
            except StopIteration:
                norm_device = None
        num_layers = len(outputs.hidden_states)
        for layer_idx, h in enumerate(outputs.hidden_states):
            h_last = h[:, last_token_index, :]
            if final_norm is not None and layer_idx != num_layers - 1:
                if norm_device is not None and h_last.device != norm_device:
                    h_last = h_last.to(norm_device)
                h_last = final_norm(h_last)
            hidden_by_layer.append(h_last.detach().to("cpu"))
        return torch.stack(hidden_by_layer, dim=1)

    @torch.no_grad()
    def forward_last_hidden_by_layer(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        last_token_index = int(attention_mask[0].sum().item()) - 1

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        return self._last_token_hidden_by_layer(outputs, last_token_index), outputs.past_key_values

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())

        for step in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._prepare_latent_vec(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past

    @torch.no_grad()
    def generate_latent_batch_with_trace(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[Tuple, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        prompt_last_token_index = int(attention_mask[0].sum().item()) - 1

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        trace = [self._last_token_hidden_by_layer(outputs, prompt_last_token_index)]

        for _ in range(latent_steps):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._prepare_latent_vec(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            trace.append(self._last_token_hidden_by_layer(outputs, 0))

        return past, torch.stack(trace, dim=1)
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._prepare_latent_vec(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings
