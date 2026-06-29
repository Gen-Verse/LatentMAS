"""
ActivationExtractor: hook-based hidden state extraction from HuggingFace models.

Registers forward hooks on transformer decoder layers to capture mean-pooled
hidden states for arbitrary subsets of layers, enabling efficient extraction
of contrastive pairs for multilingual analysis.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal capture buffer
# ---------------------------------------------------------------------------

class _HiddenStateCapture:
    """Lightweight hook-based hidden state buffer.

    Attributes
    ----------
    layer_id : int
        The layer index this capture is attached to.
    states : Optional[Tensor]
        Buffer holding the last captured hidden states.
    """

    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id
        self.states: Optional[Tensor] = None

    def hook_fn(self, module: nn.Module, inputs: tuple, output) -> None:
        """Forward hook that captures the first element of the output tuple."""
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        # Store a CPU detached copy to avoid memory leaks on GPU
        self.states = hs.detach().cpu()

    def reset(self) -> None:
        self.states = None


# ---------------------------------------------------------------------------
# Layer resolver
# ---------------------------------------------------------------------------

def _get_decoder_layers(model: PreTrainedModel) -> List[nn.Module]:
    """Robustly resolve the list of transformer decoder layers.

    Delegates to shared.model_utils.get_transformer_layers which centralises
    all architecture pattern matching across the three projects.
    """
    from shared.model_utils import get_transformer_layers
    return get_transformer_layers(model)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ActivationExtractor:
    """Extract hidden states from HuggingFace models via forward hooks.

    Parameters
    ----------
    model : PreTrainedModel
        A HuggingFace causal LM or VLM (in eval mode).
    tokenizer : PreTrainedTokenizerBase
        Corresponding tokenizer.
    device : str, optional
        Torch device.  Defaults to ``"cpu"``.
    pooling : str, optional
        How to pool over the sequence dimension:
        ``"mean"`` (default), ``"last"``, or ``"max"``.

    Examples
    --------
    >>> extractor = ActivationExtractor(model, tokenizer, device="cuda")
    >>> layer_states = extractor.extract(texts, layer_ids=[16, 20, 24])
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cpu",
        pooling: str = "mean",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.pooling = pooling
        self._decoder_layers: Optional[List[nn.Module]] = None

        # Ensure model is in eval mode
        self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        # Resolve decoder layers once
        try:
            self._decoder_layers = _get_decoder_layers(model)
            logger.info(
                "ActivationExtractor ready | n_layers=%d device=%s pooling=%s",
                len(self._decoder_layers),
                device,
                pooling,
            )
        except AttributeError as e:
            logger.warning("Could not auto-resolve layers: %s", e)

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        """Number of transformer decoder layers."""
        if self._decoder_layers is None:
            raise RuntimeError("Decoder layers not resolved.")
        return len(self._decoder_layers)

    def extract(
        self,
        texts: List[str],
        layer_ids: Optional[List[int]] = None,
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> Dict[int, Tensor]:
        """Extract mean-pooled hidden states for specified layers.

        Parameters
        ----------
        texts : List[str]
            Input texts to process.
        layer_ids : List[int], optional
            Layer indices to extract from.  Defaults to all layers.
        batch_size : int, optional
            Texts per forward pass.  Defaults to 8.
        device : str, optional
            Override device for this call.

        Returns
        -------
        Dict[int, Tensor]
            Mapping ``{layer_id: Tensor of shape (n_texts, hidden_dim)}``.
        """
        if self._decoder_layers is None:
            raise RuntimeError("Decoder layers not resolved.")

        _device = torch.device(device) if device else self.device
        all_layer_ids = layer_ids if layer_ids is not None else list(range(self.n_layers))

        # Validate layer ids
        for lid in all_layer_ids:
            if lid < 0 or lid >= self.n_layers:
                raise ValueError(
                    f"Layer id {lid} out of range [0, {self.n_layers - 1}]"
                )

        logger.info(
            "Extracting activations | n_texts=%d layers=%s batch_size=%d",
            len(texts),
            all_layer_ids,
            batch_size,
        )

        # Accumulator: layer_id -> list of pooled tensors (one per batch)
        accum: Dict[int, List[Tensor]] = {lid: [] for lid in all_layer_ids}

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]
            logger.debug(
                "Batch %d/%d (%d texts)",
                batch_start // batch_size + 1,
                (len(texts) + batch_size - 1) // batch_size,
                len(batch_texts),
            )

            with self._hooked_forward(all_layer_ids) as captures:
                # Tokenize
                enc = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(_device)
                attention_mask = enc["attention_mask"]

                with torch.no_grad():
                    self.model(**enc)

                # Collect states from each capture
                for lid, capture in captures.items():
                    if capture.states is None:
                        logger.warning("No states captured for layer %d", lid)
                        continue
                    hs = capture.states  # (batch, seq_len, hidden_dim)
                    pooled = self._pool(hs, attention_mask.cpu())
                    accum[lid].append(pooled)

        # Concatenate batches
        result: Dict[int, Tensor] = {}
        for lid in all_layer_ids:
            if accum[lid]:
                result[lid] = torch.cat(accum[lid], dim=0)
                logger.debug(
                    "Layer %d: extracted shape %s", lid, result[lid].shape
                )

        return result

    def extract_contrastive_pairs(
        self,
        en_texts: List[str],
        tgt_texts: List[str],
        layer_ids: Optional[List[int]] = None,
        batch_size: int = 8,
    ) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
        """Extract paired (English, target) hidden states.

        Parameters
        ----------
        en_texts : List[str]
            English texts.
        tgt_texts : List[str]
            Aligned target-language texts.
        layer_ids : List[int], optional
            Layers to extract.
        batch_size : int, optional
            Batch size.

        Returns
        -------
        en_states : Dict[int, Tensor]
            English hidden states per layer.
        tgt_states : Dict[int, Tensor]
            Target-language hidden states per layer.
        """
        if len(en_texts) != len(tgt_texts):
            raise ValueError(
                f"Parallel corpus size mismatch: {len(en_texts)} en vs {len(tgt_texts)} tgt"
            )

        logger.info(
            "Extracting contrastive pairs | n_pairs=%d", len(en_texts)
        )

        en_states = self.extract(en_texts, layer_ids=layer_ids, batch_size=batch_size)
        tgt_states = self.extract(tgt_texts, layer_ids=layer_ids, batch_size=batch_size)

        logger.info(
            "Contrastive extraction complete | layers=%d", len(en_states)
        )
        return en_states, tgt_states

    def extract_all_layers(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> Dict[int, Tensor]:
        """Extract hidden states from all transformer layers.

        Parameters
        ----------
        texts : List[str]
            Input texts.
        batch_size : int, optional
            Batch size.

        Returns
        -------
        Dict[int, Tensor]
            All-layer hidden states.
        """
        return self.extract(
            texts,
            layer_ids=list(range(self.n_layers)),
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _hooked_forward(
        self, layer_ids: List[int]
    ) -> Generator[Dict[int, _HiddenStateCapture], None, None]:
        """Context manager that installs and removes forward hooks."""
        captures: Dict[int, _HiddenStateCapture] = {}
        handles = []

        for lid in layer_ids:
            capture = _HiddenStateCapture(lid)
            handle = self._decoder_layers[lid].register_forward_hook(capture.hook_fn)
            captures[lid] = capture
            handles.append(handle)

        try:
            yield captures
        finally:
            for handle in handles:
                handle.remove()
            for capture in captures.values():
                capture.reset()

    def _pool(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool over the sequence dimension.

        Parameters
        ----------
        hidden_states : Tensor
            Shape ``(batch, seq_len, hidden_dim)``.
        attention_mask : Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Pooled representation, shape ``(batch, hidden_dim)``.
        """
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)

        if self.pooling == "mean":
            summed = (hidden_states * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return summed / count

        elif self.pooling == "last":
            # Last non-padding token
            lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_size = hidden_states.shape[0]
            return hidden_states[
                torch.arange(batch_size), lengths.clamp(min=0), :
            ]

        elif self.pooling == "max":
            masked_hs = hidden_states * mask + (-1e9) * (1.0 - mask)
            return masked_hs.max(dim=1).values

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
