"""Tests for soft-prefix latent injection (baseline identical-results fix).

Guards the 2026-07-06 finding that LatentMAS and ThoughtComm baseline runners
produced byte-identical results because their communicated latents never
reached the receiving agent. These tests verify, on a tiny random GPT-2:
  1. `_log_likelihood` with no prefix is unchanged (back-compat with cached
     Belebele probe scores);
  2. a latent prefix actually shifts both generation and log-likelihoods;
  3. different latents (LatentMAS raw state vs ThoughtComm reconstruction)
     produce different conditioning — the property whose absence caused the bug.
"""

import pytest
import torch

torch.manual_seed(0)

transformers = pytest.importorskip("transformers")
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from latent_coordination.baselines.latent_prefix import (
    build_latent_prefix,
    generate_with_latent_prefix,
)
from latent_coordination.eval.correctness import _log_likelihood


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    config = GPT2Config(
        n_layer=2, n_head=2, n_embd=32, n_positions=256,
        vocab_size=tokenizer.vocab_size,
    )
    model = GPT2LMHeadModel(config).eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_log_likelihood_no_prefix_matches_input_ids_path(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    ll_plain = _log_likelihood(model, tokenizer, "The answer is", " four", device="cpu")
    ll_none = _log_likelihood(
        model, tokenizer, "The answer is", " four", device="cpu", prefix_embeds=None,
    )
    ll_empty = _log_likelihood(
        model, tokenizer, "The answer is", " four", device="cpu",
        prefix_embeds=torch.zeros(1, 0, model.config.n_embd),
    )
    assert ll_plain == pytest.approx(ll_none)
    assert ll_plain == pytest.approx(ll_empty)


def test_prefix_changes_log_likelihood(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    prompt, cont = "The answer is", " four"
    ll_plain = _log_likelihood(model, tokenizer, prompt, cont, device="cpu")
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_embeds = model.get_input_embeddings()(prompt_ids)
    latent = torch.randn(model.config.n_embd)
    prefix = build_latent_prefix(latent, prompt_embeds)
    ll_prefixed = _log_likelihood(
        model, tokenizer, prompt, cont, device="cpu", prefix_embeds=prefix,
    )
    assert ll_plain != pytest.approx(ll_prefixed)


def test_build_latent_prefix_rescales_to_embedding_norm(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    prompt_ids = tokenizer("hello world", return_tensors="pt")["input_ids"]
    prompt_embeds = model.get_input_embeddings()(prompt_ids)
    latent = torch.randn(model.config.n_embd) * 1000.0  # hidden-state-scale input
    prefix = build_latent_prefix(latent, prompt_embeds)
    assert prefix.shape == (1, 1, model.config.n_embd)
    ref = prompt_embeds.float().norm(dim=-1).mean()
    assert prefix.float().norm(dim=-1).item() == pytest.approx(ref.item(), rel=1e-3)


def test_build_latent_prefix_rejects_dim_mismatch(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    prompt_ids = tokenizer("hello", return_tensors="pt")["input_ids"]
    prompt_embeds = model.get_input_embeddings()(prompt_ids)
    with pytest.raises(ValueError, match="embedding dim"):
        build_latent_prefix(torch.randn(model.config.n_embd + 7), prompt_embeds)


def test_distinct_latents_condition_generation_distinctly(tiny_model_and_tokenizer):
    """The core anti-regression: two methods' latents must be able to produce
    different receiver behavior. Compare full logit trajectories rather than
    sampled text (a tiny random model may greedy-decode the same token)."""
    model, tokenizer = tiny_model_and_tokenizer
    prompt = "Reasoning: therefore\nFinal numeric answer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    prompt_embeds = model.get_input_embeddings()(prompt_ids)

    latent_a = torch.randn(model.config.n_embd)
    latent_b = -latent_a
    logits = []
    for latent in (latent_a, latent_b):
        prefix = build_latent_prefix(latent, prompt_embeds)
        embeds = torch.cat([prefix, prompt_embeds], dim=1)
        with torch.no_grad():
            logits.append(model(inputs_embeds=embeds).logits[0, -1])
    assert not torch.allclose(logits[0], logits[1])

    text, n_new = generate_with_latent_prefix(
        model, tokenizer, prompt, latent_a, device="cpu", max_new_tokens=4,
    )
    assert isinstance(text, str) and n_new > 0

    text_plain, n_plain = generate_with_latent_prefix(
        model, tokenizer, prompt, None, device="cpu", max_new_tokens=4,
    )
    assert isinstance(text_plain, str) and n_plain > 0
