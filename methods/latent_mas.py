from typing import Dict, List, Optional, Tuple

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout
import torch
import argparse
from vllm import SamplingParams
import pdb
from tqdm import tqdm

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = getattr(args, "custom_agents", None) or default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False
        self.first_agent_text = bool(getattr(args, "first_agent_text", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=args.max_new_tokens,
        )
        self.task = args.task

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        agent_pbar = tqdm(self.agents, desc="Agents", unit="agent")
        for agent_idx, agent in enumerate(agent_pbar):
            agent_pbar.set_description(f"Agent: {agent.name} ({agent.role})")

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]


            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            # Check if this is the first agent and should generate text
            is_first_agent = (agent_idx == 0)
            is_last_agent = (agent_idx == len(self.agents) - 1)
            should_generate_text = is_first_agent and self.first_agent_text

            if not is_last_agent:
                # First agent can optionally generate text for structured reasoning (graphs, etc.)
                if should_generate_text:
                    # Generate text for first agent
                    if self.args.think:
                        first_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                    else:
                        first_agent_prompts = prompts

                    first_encoded = self.model.tokenizer(
                        first_agent_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    first_ids = first_encoded["input_ids"].to(self.model.device)
                    first_mask = first_encoded["attention_mask"].to(self.model.device)
                    first_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(first_ids, first_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        first_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                    # Generate text output
                    generated_batch, past_kv = self.model.generate_text_batch(
                        first_ids,
                        first_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=past_kv,
                    )

                    # NOTE: We keep the EOS token in the KV cache
                    # For chat models, EOS is typically <|im_end|> which serves as the turn delimiter
                    # Structure: <|im_start|>assistant [text] <|im_end|> <|im_start|>user [next]
                    # Removing it would break the chat template structure

                    for idx in range(batch_size):
                        text_out = generated_batch[idx].strip()
                        mask = first_mask[idx].bool()
                        trimmed_ids = first_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": first_agent_prompts[idx],
                                "input_ids": trimmed_ids,
                                "input_tokens": first_tokens_batch[idx],
                                "output": text_out,
                            }
                        )
                else:
                    # Standard latent generation for non-first agents or when flag not set
                    prev_past_len = _past_length(past_kv)

                    if self.args.think:
                        wrapped_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                    else:
                        wrapped_prompts = prompts

                    wrapped_encoded = self.model.tokenizer(
                        wrapped_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                    wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                    wrapped_tokens_batch: List[List[str]] = []
                    for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                        active_ids = ids_row[mask_row.bool()].tolist()
                        wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                    past_kv = self.model.generate_latent_batch(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )
                    if self.sequential_info_only or self.latent_only:
                        new_past_len = _past_length(past_kv)
                        tokens_added = new_past_len - prev_past_len
                        tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                        past_kv = self._truncate_past(past_kv, tokens_to_keep)

                    for idx in range(batch_size):
                        mask = wrapped_mask[idx].bool()
                        trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                        agent_traces[idx].append(
                            {
                                "name": agent.name,
                                "role": agent.role,
                                "input": wrapped_prompts[idx],
                                "input_ids": trimmed_ids,
                                "input_tokens": wrapped_tokens_batch[idx],
                                "latent_steps": self.latent_steps,
                                "output": "",
                            }
                        )
            else:
                # Last agent: Generate final text output
                past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                        final_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    final_agent_prompts = prompts

                final_agent_encoded = self.model.tokenizer(
                    final_agent_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                final_agent_ids = final_agent_encoded["input_ids"].to(self.model.device)
                final_agent_mask = final_agent_encoded["attention_mask"].to(self.model.device)
                final_agent_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(final_agent_ids, final_agent_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    final_agent_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                generated_batch, _ = self.model.generate_text_batch(
                    final_agent_ids,
                    final_agent_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = final_agent_mask[idx].bool()
                    trimmed_ids = final_agent_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": final_agent_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": final_agent_tokens_batch[idx],
                            "output": final_text,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        agent_pbar = tqdm(self.agents, desc="Agents", unit="agent")
        for agent_idx, agent in enumerate(agent_pbar):
            agent_pbar.set_description(f"Agent: {agent.name} ({agent.role})")

            is_last_agent = (agent_idx == len(self.agents) - 1)

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args)
                    for item in items
                ]

            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if not is_last_agent:
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = self._truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                # Last agent: Generate final text output
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)

                if self.args.think:
                    final_agent_prompts = [f"{prompt}{self.args.think}" for prompt in prompts]
                else:
                    final_agent_prompts = prompts

                final_agent_encoded = self.model.tokenizer(
                    final_agent_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                final_agent_encoded_ids = final_agent_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(final_agent_encoded_ids).squeeze(0).to(self.vllm_device)

                # assert Qwen model
                #assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position
                len_of_left = []
                for p in final_agent_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones 
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                
                # pdb.set_trace()              
                
                # Use vLLM 
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb 
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": final_agent_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]