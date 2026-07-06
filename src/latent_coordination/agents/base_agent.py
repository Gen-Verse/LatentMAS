"""
BaseAgent: Abstract base class for all multi-agent system participants.

Provides:
    - Lazy model + tokenizer loading
    - Hidden state extraction (with optional layer selection)
    - Latent-state injection + generation
    - Async-compatible process() interface
    - Comprehensive timing and logging instrumentation

Concrete specialisations (TranslationAgent, ReasoningAgent, SafetyAgent)
override the abstract ``process()`` method.
"""


import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)

# Try importing HuggingFace; gracefully degrade for unit-testing without models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logger.warning("transformers not installed; agent model loading will fail at runtime.")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

AgentRole = Literal["translation", "reasoning", "safety", "orchestrator"]


@dataclass
class AgentConfig:
    """Configuration for a BaseAgent instance.

    Attributes:
        agent_id: Unique agent identifier string.
        model_id: HuggingFace model ID (e.g. 'meta-llama/Llama-3-8B').
        role: Functional role of this agent.
        device: PyTorch device string.
        max_new_tokens: Max tokens to generate per call.
        hidden_dim: Model hidden size (used for adapter registration).
        default_layer_ids: Layer indices to extract hidden states from.
            If None, defaults to all layers.
        load_in_8bit: Whether to load model in 8-bit quantisation.
        load_in_4bit: Whether to load model in 4-bit quantisation.
        trust_remote_code: Passed to from_pretrained.
        dtype: Torch dtype string for model loading ('float16', 'bfloat16', 'float32').
        latent_transfer_layer: Decoder-layer index whose hidden states are
            captured during generation and handed to the next agent (wired from
            ``communication.latent_transfer_layer`` in configs/*.yaml).
        max_time_s: Wall-clock budget (seconds) for a single ``generate()``
            call, wired from ``orchestration.timeout_per_agent_s`` in
            configs/*.yaml via transformers' ``max_time`` stopping criterion.
            ``None`` disables the cap.
    """

    agent_id: str
    model_id: str
    role: AgentRole = "reasoning"
    device: str = "cpu"
    max_new_tokens: int = 512
    hidden_dim: int = 4096
    default_layer_ids: Optional[List[int]] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    dtype: str = "float32"
    latent_transfer_layer: int = -1
    max_time_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """A single task assigned to an agent.

    Attributes:
        task_id: Unique task identifier.
        query: Natural language query / instruction.
        context: Optional additional context or conversation history -- this is fed
            verbatim into every specialized agent's prompt (see specialized_agents.py's
            *_build_*_prompt methods and SafetyAgent.process), so it must never carry
            the gold/reference answer. Benchmark loaders that need a gold answer for
            scoring must put it in `reference`, not here (a prior bug did exactly this
            for FLORES+ -- see `reference`'s docstring).
        latent_state: Optional incoming latent tensor from another agent
            (shape depends on the agent's hidden_dim).
        target_language: ISO 639-1 code for translation tasks (e.g. 'id').
        metadata: Arbitrary extra information (source_agent, priority, etc.).
        reference: Optional gold/reference answer, for scoring only (e.g.
            eval.benchmark_runner::MultiAgentBenchmarkRunner._compute_translation_quality
            reads this to score chrf/xcomet/cometkiwi against the FLORES+ gold
            translation). Never read by any agent's prompt-building code -- if it were,
            every agent would receive the answer key as part of its own input, which is
            exactly the leak this field was added to fix (found 2026-07-02: `context`
            was previously overloaded to carry the FLORES+ gold translation, and both
            TranslationAgent and ReasoningAgent's prompts include `task.context`
            verbatim, so the first agent in every comm-mode was handed the answer).
    """

    task_id: str
    query: str
    context: str = ""
    latent_state: Optional[Tensor] = None
    target_language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    reference: Optional[str] = None


@dataclass
class AgentResponse:
    """Response produced by an agent after processing a task.

    Attributes:
        task_id: ID of the task this is responding to.
        agent_id: ID of the agent that produced this response.
        output_text: Generated text output.
        latent_state: Hidden state tensor extracted from the last forward pass,
            shape (1, seq_len, hidden_dim) or (1, hidden_dim).
        confidence: Optional scalar confidence estimate in [0, 1].
        metadata: Arbitrary response metadata.
        elapsed_ms: Wall-clock processing time in milliseconds.
    """

    task_id: str
    agent_id: str
    output_text: str
    latent_state: Optional[Tensor] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Abstract base class for all multi-agent system participants.

    Subclasses must implement :py:meth:`process` to handle :class:`AgentTask`
    objects and return :class:`AgentResponse` objects.

    Lazy loading:
        The underlying HuggingFace model and tokenizer are only loaded when
        :py:meth:`_ensure_model_loaded` is first called (or when any method
        that needs them is invoked).  This allows creating agents without
        immediately consuming GPU memory.

    Args:
        config: :class:`AgentConfig` with all agent settings.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.agent_id = config.agent_id
        self.role = config.role
        self._device = torch.device(config.device)
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._is_loaded = False
        self._call_count = 0
        self._total_elapsed_ms = 0.0

        logger.info(
            "BaseAgent '%s' (role=%s, model=%s) created [lazy loading]",
            config.agent_id,
            config.role,
            config.model_id,
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, task: AgentTask) -> AgentResponse:
        """Process a task and return a response.

        Subclasses implement this method to apply their specialised logic
        (translation, reasoning, safety checking, etc.).

        Args:
            task: The :class:`AgentTask` to process.

        Returns:
            :class:`AgentResponse` with the result.
        """
        ...

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def async_process(self, task: AgentTask) -> AgentResponse:
        """Async-compatible wrapper around :py:meth:`process`.

        Runs the synchronous ``process`` call in a thread-pool executor so it
        doesn't block the event loop, enabling concurrent agent execution via
        ``asyncio.gather``.

        Args:
            task: The :class:`AgentTask` to process.

        Returns:
            :class:`AgentResponse` from the underlying ``process`` call.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, task)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazily load the HuggingFace model and tokenizer.

        Called internally before any forward pass.  Raises ``ImportError``
        if ``transformers`` is not installed.
        """
        if self._is_loaded:
            return

        if not _HF_AVAILABLE:
            raise ImportError(
                "transformers is required for agent model loading. "
                "Install it with: pip install transformers"
            )

        cfg = self.config
        logger.info(
            "Loading model '%s' for agent '%s' on %s ...",
            cfg.model_id,
            cfg.agent_id,
            cfg.device,
        )
        t0 = time.perf_counter()

        # Single source of truth: the shared accelerate-backed loader handles V100-safe
        # dtype (no bf16), bitsandbytes quantisation, and device placement.
        from shared.model_loader import ModelLoadSpec, load_model_and_tokenizer

        spec = ModelLoadSpec(
            model_id=cfg.model_id,
            device=cfg.device,
            dtype=cfg.dtype,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            output_hidden_states=True,          # agents need hidden states for latent transfer
            trust_remote_code=cfg.trust_remote_code,
        )
        self._model, self._tokenizer = load_model_and_tokenizer(spec)
        self._is_loaded = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Model '%s' loaded in %.1f ms", cfg.model_id, elapsed
        )

    # ------------------------------------------------------------------
    # Hidden state extraction
    # ------------------------------------------------------------------

    def extract_hidden_states(
        self,
        text: str,
        layer_ids: Optional[List[int]] = None,
    ) -> Dict[int, Tensor]:
        """Extract intermediate hidden states from the model for given text.

        Args:
            text: Input text string to tokenise and forward through the model.
            layer_ids: Specific transformer layer indices to extract from.
                If None, uses ``config.default_layer_ids`` or all layers.

        Returns:
            Dict mapping layer index -> hidden state tensor of shape
            (1, seq_len, hidden_dim).

        Example::

            states = agent.extract_hidden_states("Hello world", layer_ids=[-1])
        """
        self._ensure_model_loaded()

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        # outputs.hidden_states: tuple of (1, seq_len, hidden_dim) per layer
        all_hidden = outputs.hidden_states  # tuple[Tensor]
        n_layers = len(all_hidden)

        if layer_ids is None:
            layer_ids = self.config.default_layer_ids
        if layer_ids is None:
            layer_ids = list(range(n_layers))

        result: Dict[int, Tensor] = {}
        for lid in layer_ids:
            if -n_layers <= lid < n_layers:
                result[lid] = all_hidden[lid].detach()
            else:
                logger.warning("Layer %d out of range (model has %d layers)", lid, n_layers)

        return result

    # ------------------------------------------------------------------
    # Latent injection + generation
    # ------------------------------------------------------------------

    def _decoder_layers(self) -> List[Any]:
        """Return the model's decoder layer list (LLaMA-family layout)."""
        return list(self._model.model.layers)

    def _resolve_layer_index(self, layer: int, n_layers: int) -> int:
        """Clamp a (possibly negative) layer index into [0, n_layers - 1]."""
        idx = layer if layer >= 0 else n_layers + layer
        return max(0, min(idx, n_layers - 1))

    def _make_injection_hook(self, hidden_states: Tensor):
        """Build a prefill-only injection hook replacing a layer's output.

        The hook must fire ONLY on the prefill (first) forward pass: during
        incremental decoding with a KV cache the layer output has seq_len == 1,
        and re-injecting on every step overwrote each newly generated token's
        hidden state with the same first injected vector — degenerating the
        whole generation instead of conditioning it once on the transferred
        state.
        """
        target_states = hidden_states.to(self._device)
        _injected_once = [False]

        def _hook_fn(module, input, output):
            if _injected_once[0]:
                return output
            # output is typically a tuple; first element is the hidden states
            if isinstance(output, tuple):
                _injected_once[0] = True
                hs = output[0]
                B, L, D = hs.shape
                # target_states comes from UniversalLatentHub's decode (a plain
                # float32 LatentAdapter) while the model itself typically runs in
                # float16/8bit -- injecting without matching dtype crashes the next
                # layer's matmul (e.g. lm_head) with a float/Half mismatch. Cast to
                # the surrounding layer's actual runtime dtype every call rather
                # than once outside the hook, so this is correct regardless of the
                # model's precision.
                inj_states = target_states.to(dtype=hs.dtype)
                inj_B, inj_L, inj_D = inj_states.shape
                # Align sequence lengths by padding / truncating
                if inj_L >= L:
                    injected = inj_states[:B, :L, :D]
                else:
                    injected = torch.cat(
                        [inj_states[:B, :, :D],
                         hs[:B, inj_L:, :D]], dim=1
                    )
                return (injected,) + output[1:]
            return output

        return _hook_fn

    def generate_and_capture(
        self,
        input_text: str,
        latent_state: Optional[Tensor] = None,
        injection_layer: int = -1,
        capture_layer: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 0.7,
    ) -> Tuple[str, Optional[Tensor]]:
        """Generate text while capturing GENERATION-TIME hidden states.

        This is the latent hand-off producer for the text-free channel: the
        states transferred to the next agent are the ones the model actually
        computed while producing its answer (one vector per decode step at
        ``capture_layer``), NOT a re-encoding of the answer text. Re-encoding
        the agent's own output text (the previous behaviour) discards the
        reasoning trajectory and reduces the "latent channel" to a lossy text
        round-trip — dev_doc.md §9 gap 5.

        Args:
            input_text: Prompt to condition generation on.
            latent_state: Optional incoming latent to inject at
                ``injection_layer`` before generation (prefill-only).
            injection_layer: Layer index for injection (used only when
                ``latent_state`` is not None).
            capture_layer: Layer whose hidden states are captured. Defaults to
                ``config.latent_transfer_layer``.
            max_new_tokens: Override ``config.max_new_tokens``.
            do_sample, temperature: Sampling controls (greedy by default for
                reproducible benchmark runs).

        Returns:
            Tuple ``(generated_text, captured_states)`` where
            ``captured_states`` has shape (1, 1 + n_generated_steps, hidden_dim)
            — the final prompt position from prefill plus one vector per
            generated token — or None if generation produced no forward pass.
        """
        self._ensure_model_loaded()
        max_new = max_new_tokens or self.config.max_new_tokens
        # getattr: AgentConfig instances restored from pre-field checkpoints
        # lack the attribute entirely.
        cap_layer = (
            capture_layer if capture_layer is not None
            else getattr(self.config, "latent_transfer_layer", -1)
        )

        layers = self._decoder_layers()
        n_layers = len(layers)
        handles = []
        captured_chunks: List[Tensor] = []

        def _capture_hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            # Prefill contributes the final prompt position; each decode step
            # contributes its single new position. Keeping only the last
            # position per forward pass is what makes this memory-light
            # (D floats per step instead of the full L x D x n_layers cube
            # `generate(output_hidden_states=True)` would allocate).
            captured_chunks.append(hs[:, -1:, :].detach().to(torch.float32).cpu())
            return output

        # Injection hook FIRST: forward hooks chain in registration order, so
        # when both hooks land on the same layer the capture sees the
        # post-injection (effective) states, not the overwritten ones.
        if latent_state is not None:
            inj_idx = self._resolve_layer_index(injection_layer, n_layers)
            handles.append(
                layers[inj_idx].register_forward_hook(
                    self._make_injection_hook(latent_state)
                )
            )
        cap_idx = self._resolve_layer_index(cap_layer, n_layers)
        handles.append(layers[cap_idx].register_forward_hook(_capture_hook))

        try:
            inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)
            gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new, "do_sample": do_sample}
            if do_sample:
                gen_kwargs["temperature"] = temperature
            if self.config.max_time_s is not None:
                gen_kwargs["max_time"] = float(self.config.max_time_s)
            with torch.no_grad():
                out_ids = self._model.generate(**inputs, **gen_kwargs)
        finally:
            for h in handles:
                h.remove()

        generated_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        captured = torch.cat(captured_chunks, dim=1) if captured_chunks else None
        return text, captured

    def inject_latent_and_generate(
        self,
        hidden_states: Tensor,
        input_text: str,
        injection_layer: int = -1,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = False,
    ) -> str:
        """Generate text using modified (injected) hidden states.

        Performs a two-step process:
        1. Encode ``input_text`` to get the model's key-value cache up to
           the injection layer.
        2. Replace the hidden state at ``injection_layer`` with the provided
           ``hidden_states`` tensor and continue generation.

        Note:
            This is an approximation: for simplicity the hidden states are
            injected by modifying the model's last-layer hidden states via
            hooks.  A rigorous implementation would require architecture-
            specific layer patching.

        Args:
            hidden_states: Tensor to inject, shape (1, seq_len, hidden_dim).
            input_text: Prompt text to tokenise.
            injection_layer: Layer index at which to inject states.
            max_new_tokens: Override config ``max_new_tokens``.
            temperature: Sampling temperature.
            do_sample: Whether to use sampling (vs greedy).

        Returns:
            Decoded generated text string.
        """
        self._ensure_model_loaded()
        max_new = max_new_tokens or self.config.max_new_tokens

        # Prefill-only injection hook (see _make_injection_hook for the
        # decode-step re-injection bug this guards against).
        layers = self._decoder_layers()  # works for LLaMA-family
        idx = self._resolve_layer_index(injection_layer, len(layers))
        hook_handle = layers[idx].register_forward_hook(
            self._make_injection_hook(hidden_states)
        )

        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new, "do_sample": do_sample}
        if do_sample:
            # Only pass temperature when sampling — HF warns (and ignores it) in
            # greedy mode. Benchmark evals default to greedy for reproducibility.
            gen_kwargs["temperature"] = temperature
        if self.config.max_time_s is not None:
            gen_kwargs["max_time"] = float(self.config.max_time_s)
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, **gen_kwargs)
        hook_handle.remove()
        # Decode only the newly generated tokens
        generated_ids = out_ids[0, inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _start_timer(self) -> float:
        """Start a timing measurement."""
        return time.perf_counter()

    def _stop_timer(self, t0: float) -> float:
        """Return elapsed milliseconds since t0."""
        return (time.perf_counter() - t0) * 1000.0

    def _build_response(
        self,
        task: AgentTask,
        output_text: str,
        latent_state: Optional[Tensor],
        elapsed_ms: float,
        confidence: Optional[float] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Convenience factory for building AgentResponse objects."""
        self._call_count += 1
        self._total_elapsed_ms += elapsed_ms
        meta = {"call_count": self._call_count, "role": self.role}
        if extra_meta:
            meta.update(extra_meta)
        return AgentResponse(
            task_id=task.task_id,
            agent_id=self.agent_id,
            output_text=output_text,
            latent_state=latent_state,
            confidence=confidence,
            metadata=meta,
            elapsed_ms=elapsed_ms,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative runtime statistics for this agent."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "call_count": self._call_count,
            "total_elapsed_ms": self._total_elapsed_ms,
            "avg_elapsed_ms": (
                self._total_elapsed_ms / self._call_count if self._call_count > 0 else 0.0
            ),
            "is_loaded": self._is_loaded,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id='{self.agent_id}', role='{self.role}', "
            f"model='{self.config.model_id}', loaded={self._is_loaded})"
        )
