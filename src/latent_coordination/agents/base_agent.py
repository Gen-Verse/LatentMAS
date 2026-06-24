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

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor

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
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """A single task assigned to an agent.

    Attributes:
        task_id: Unique task identifier.
        query: Natural language query / instruction.
        context: Optional additional context or conversation history.
        latent_state: Optional incoming latent tensor from another agent
            (shape depends on the agent's hidden_dim).
        target_language: ISO 639-1 code for translation tasks (e.g. 'id').
        metadata: Arbitrary extra information (source_agent, priority, etc.).
    """

    task_id: str
    query: str
    context: str = ""
    latent_state: Optional[Tensor] = None
    target_language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    def inject_latent_and_generate(
        self,
        hidden_states: Tensor,
        input_text: str,
        injection_layer: int = -1,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
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

        # Register a forward hook that replaces hidden states at target layer
        _hook_output: List[Tensor] = []
        target_states = hidden_states.to(self._device)

        def _hook_fn(module, input, output):
            # output is typically a tuple; first element is the hidden states
            if isinstance(output, tuple):
                hs = output[0]
                B, L, D = hs.shape
                inj_B, inj_L, inj_D = target_states.shape
                # Align sequence lengths by padding / truncating
                if inj_L >= L:
                    injected = target_states[:B, :L, :D]
                else:
                    injected = torch.cat(
                        [target_states[:B, :, :D],
                         hs[:B, inj_L:, :D]], dim=1
                    )
                return (injected,) + output[1:]
            return output

        layers = list(self._model.model.layers)  # works for LLaMA-family
        n_layers = len(layers)
        idx = injection_layer if injection_layer >= 0 else n_layers + injection_layer
        idx = max(0, min(idx, n_layers - 1))
        hook_handle = layers[idx].register_forward_hook(_hook_fn)

        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temperature,
                do_sample=do_sample,
            )
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
