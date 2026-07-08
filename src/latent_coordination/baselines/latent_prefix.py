"""Soft-prefix latent injection for the training-free MAS baselines.

Both baseline runners (LatentMAS, ThoughtComm) previously computed their
communicated latent and then discarded it, conditioning Agent 2 on Agent 1's
*text* only — which made the two "methods" byte-identical single-model prompt
chains (identical n_correct and mean_token_cost across all MGSM languages,
found 2026-07-06). This module makes the communicated vector actually reach
the receiving agent: the latent is rescaled to the prompt's input-embedding
norm and prepended as soft prefix token(s) via ``inputs_embeds``, so it
influences every generated/scored token without custom forward hooks.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

__author__ = "Himon Thakur"
__license__ = "Apache 2.0"

logger = logging.getLogger(__name__)


def build_latent_prefix(latent: Tensor, prompt_embeds: Tensor) -> Tensor:
    """Shape/scale a communicated latent into soft prefix embeddings.

    Args:
        latent: Communicated vector(s), shape (D,), (B, D) or (B, K, D); D must
            equal the receiver's embedding width (homogeneous baselines).
        prompt_embeds: The receiver prompt's input embeddings, shape (1, T, D) —
            used as the norm reference so the prefix lives on the same scale as
            real token embeddings (hidden-state norms are orders of magnitude
            larger than input-embedding norms).

    Returns:
        Prefix embeddings of shape (1, K, D), dtype/device of ``prompt_embeds``.
    """
    if latent.dim() == 1:
        latent = latent.unsqueeze(0)
    if latent.dim() == 2:
        latent = latent.unsqueeze(1)  # (B, 1, D)
    prefix = latent[:1].to(device=prompt_embeds.device, dtype=torch.float32)
    d_model = prompt_embeds.shape[-1]
    if prefix.shape[-1] != d_model:
        raise ValueError(
            f"Latent dim {prefix.shape[-1]} != receiver embedding dim {d_model}; "
            "homogeneous-baseline injection requires matching widths."
        )
    ref_norm = prompt_embeds.float().norm(dim=-1).mean()
    prefix_norm = prefix.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    prefix = prefix / prefix_norm * ref_norm
    return prefix.to(dtype=prompt_embeds.dtype)


def generate_with_latent_prefix(
    model,
    tokenizer,
    prompt: str,
    latent: Optional[Tensor],
    device: str,
    max_new_tokens: int,
) -> Tuple[str, int]:
    """Greedy-generate conditioned on ``[latent prefix] + prompt`` embeddings.

    With ``latent=None`` this degrades to plain text-conditioned generation
    (used for the documented heterogeneous-failure path). Returns
    ``(output_text, n_new_tokens)`` like the runners' ``_generate_text``.
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    if latent is None:
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        n_new = out_ids.shape[1] - input_ids.shape[1]
        text = tokenizer.decode(out_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        return text, int(n_new)

    embed_layer = model.get_input_embeddings()
    with torch.no_grad():
        prompt_embeds = embed_layer(input_ids)  # (1, T, D)
        prefix = build_latent_prefix(latent, prompt_embeds)  # (1, K, D)
        embeds = torch.cat([prefix, prompt_embeds], dim=1)
        mask = torch.ones(embeds.shape[:2], dtype=attention_mask.dtype, device=embeds.device)
        # With inputs_embeds and no input_ids, generate() returns ONLY the new tokens.
        out_ids = model.generate(
            inputs_embeds=embeds,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text, int(out_ids.shape[1])
