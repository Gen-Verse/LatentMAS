"""
Logit Lens — Hidden State Mapping.

Projects the hidden state at each transformer layer through the model's final
layer normalisation and LM head to obtain a vocabulary probability distribution.
Comparing these per-layer distributions for English vs non-English inputs reveals
the exact layers where probability mass collapses from the target language toward
English — the mechanistic signature of latent language bias.

Reference: nostalgebraist (2020) "interpreting GPT: the logit lens"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from mrre_drift.models.layers import get_transformer_layers
    from mrre_drift.utils.capture import HiddenStateCapture
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# LM-head component discovery
# ---------------------------------------------------------------------------

_NORM_ATTRS = ("model.norm", "model.language_model.norm", "model.final_layernorm", "transformer.ln_f", "norm")
_HEAD_ATTRS = ("lm_head", "embed_out", "output")


def _find_module(model: "nn.Module", paths: Tuple[str, ...]) -> Optional["nn.Module"]:
    for path in paths:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def get_lm_head_components(
    model: "nn.Module",
) -> Tuple[Optional["nn.Module"], Optional["nn.Module"]]:
    """Return (final_layer_norm, lm_head) for any supported causal LM architecture."""
    return (
        _find_module(model, _NORM_ATTRS),
        _find_module(model, _HEAD_ATTRS),
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LayerLensResult:
    """Logit-lens probe result at one transformer layer."""
    layer_id: int
    top_tokens: List[Tuple[str, float]]  # (token_str, probability), top-k
    english_mass: float                   # sum P(token) for tokens in english_token_ids
    target_mass: float                    # sum P(token) for tokens in target_token_ids
    entropy: float                        # H(distribution) in nats


@dataclass
class LogitLensScan:
    """Full per-layer logit-lens scan for a single input text."""
    text: str
    layer_results: List[LayerLensResult] = field(default_factory=list)

    @property
    def english_mass_by_layer(self) -> Dict[int, float]:
        return {r.layer_id: r.english_mass for r in self.layer_results}

    @property
    def target_mass_by_layer(self) -> Dict[int, float]:
        return {r.layer_id: r.target_mass for r in self.layer_results}

    def collapse_onset_layer(self, threshold: float = 0.5) -> Optional[int]:
        for r in self.layer_results:
            if r.english_mass >= threshold:
                return r.layer_id
        return None

    def peak_english_layer(self) -> int:
        return max(self.layer_results, key=lambda r: r.english_mass).layer_id


# ---------------------------------------------------------------------------
# LogitLens
# ---------------------------------------------------------------------------

class LogitLens:
    """
    Per-layer language probability probe for causal transformer LMs.

    Parameters
    ----------
    model             : causal LM (HuggingFace AutoModelForCausalLM)
    tokenizer         : matching tokenizer
    english_token_ids : vocabulary indices considered "English"
                        (defaults to ASCII-only tokens if not provided)
    top_k             : number of top-probability tokens to record per layer
    device            : device string
    """

    def __init__(
        self,
        model,
        tokenizer,
        english_token_ids: Optional[List[int]] = None,
        top_k: int = 10,
        device: str = "cpu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required to use LogitLens")
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.device = device

        final_norm, lm_head = get_lm_head_components(model)
        if lm_head is None:
            raise RuntimeError(
                f"Cannot locate lm_head in {type(model).__name__}. "
                f"Tried: {_HEAD_ATTRS}"
            )
        self._final_norm = final_norm
        self._lm_head = lm_head
        self._english_ids = (
            english_token_ids if english_token_ids is not None
            else self._default_english_ids()
        )

    def _default_english_ids(self) -> List[int]:
        ids = []
        try:
            vocab = self.tokenizer.get_vocab()
            for tok, idx in vocab.items():
                if tok is None:
                    continue
                try:
                    tok.encode("ascii")
                    ids.append(idx)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
        except Exception:
            for i in range(self.tokenizer.vocab_size):
                tok = self.tokenizer.convert_ids_to_tokens(i)
                if tok is None:
                    continue
                try:
                    tok.encode("ascii")
                    ids.append(i)
                except (UnicodeEncodeError, UnicodeDecodeError):
                    pass
        return ids

    def _project(self, hidden: "torch.Tensor") -> "torch.Tensor":
        dtype = next(self._lm_head.parameters()).dtype if list(self._lm_head.parameters()) else hidden.dtype
        h = hidden.to(dtype)
        if self._final_norm is not None:
            with torch.no_grad():
                h = self._final_norm(h)
        with torch.no_grad():
            logits = self._lm_head(h)
        return logits

    def scan(
        self,
        text: str,
        target_token_ids: Optional[List[int]] = None,
    ) -> LogitLensScan:
        """Run a full logit-lens scan over all transformer layers for `text`."""
        layers = get_transformer_layers(self.model)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        scan_result = LogitLensScan(text=text)

        with torch.no_grad(), HiddenStateCapture(layers) as cap:
            self.model(**inputs)

        english_ids_t = torch.tensor(self._english_ids, device=self.device)
        target_ids_t = (
            torch.tensor(target_token_ids, device=self.device)
            if target_token_ids else None
        )

        for pos, layer in enumerate(layers):
            hidden = cap.states[pos]  # (1, hidden_dim)
            logits = self._project(hidden)
            probs = F.softmax(logits.squeeze(0), dim=-1)

            topk_probs, topk_ids = torch.topk(probs, self.top_k)
            top_tokens = [
                (self.tokenizer.convert_ids_to_tokens(int(tid)), float(p))
                for tid, p in zip(topk_ids, topk_probs)
            ]

            en_mass = float(probs[english_ids_t].sum())
            tgt_mass = float(probs[target_ids_t].sum()) if target_ids_t is not None else 0.0

            clamped = probs.clamp(min=1e-10)
            entropy = float(-torch.sum(clamped * clamped.log()))

            scan_result.layer_results.append(LayerLensResult(
                layer_id=pos,
                top_tokens=top_tokens,
                english_mass=en_mass,
                target_mass=tgt_mass,
                entropy=entropy,
            ))

        return scan_result

    def scan_batch(
        self,
        texts: Sequence[str],
        target_token_ids: Optional[List[int]] = None,
    ) -> List[LogitLensScan]:
        return [self.scan(t, target_token_ids) for t in texts]

    def mean_scan(
        self,
        texts: Sequence[str],
        target_token_ids: Optional[List[int]] = None,
    ) -> LogitLensScan:
        """Average per-layer english_mass across all input texts."""
        scans = self.scan_batch(texts, target_token_ids)
        if not scans:
            raise ValueError("texts must not be empty")

        n_layers = len(scans[0].layer_results)
        avg_results: List[LayerLensResult] = []

        for pos in range(n_layers):
            en_masses = [s.layer_results[pos].english_mass for s in scans]
            tgt_masses = [s.layer_results[pos].target_mass for s in scans]
            entropies = [s.layer_results[pos].entropy for s in scans]
            top_tokens = scans[0].layer_results[pos].top_tokens

            avg_results.append(LayerLensResult(
                layer_id=pos,
                top_tokens=top_tokens,
                english_mass=sum(en_masses) / len(en_masses),
                target_mass=sum(tgt_masses) / len(tgt_masses),
                entropy=sum(entropies) / len(entropies),
            ))

        return LogitLensScan(text=f"[mean over {len(texts)} texts]", layer_results=avg_results)
