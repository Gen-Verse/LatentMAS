"""Tests for the gated generation-backend resolver.

These do not require a GPU or vLLM install — they exercise the capability gate and the
HF fallback path, which is the behavior that matters on V100/sm_70.
"""

import importlib

import pytest

from shared.generation_backend import resolve_backend, vllm_supported

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


def _make_vllm_unimportable(monkeypatch):
    """Force `import vllm` to fail regardless of whether it's actually installed."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ImportError("vllm forced-unavailable for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_vllm_unsupported_without_install(monkeypatch):
    # When vllm can't be imported → never "supported", regardless of GPU.
    _make_vllm_unimportable(monkeypatch)
    assert vllm_supported("cuda:0") is False
    assert vllm_supported("cpu") is False


def test_force_flag_still_requires_import(monkeypatch):
    # SRE_FORCE_VLLM bypasses the *capability* check but not a missing/broken vllm import.
    _make_vllm_unimportable(monkeypatch)
    monkeypatch.setenv("SRE_FORCE_VLLM", "1")
    assert vllm_supported("cuda:0") is False


def test_resolve_backend_falls_back_to_hf(monkeypatch):
    # Requesting vllm on an unsupported box must NOT raise — it falls back to HF.
    captured = {}

    class _FakeSpec:
        def __init__(self, **kw):
            captured.update(kw)

    class _FakeHF:
        name = "hf"

        @classmethod
        def from_spec(cls, spec):
            return cls()

    import shared.generation_backend as gb
    import shared.model_loader as ml
    monkeypatch.setattr(ml, "ModelLoadSpec", _FakeSpec)
    monkeypatch.setattr(gb, "HFBackend", _FakeHF)

    backend = resolve_backend("vllm", model_id="dummy/model", device="cuda:0")
    assert backend.name == "hf"
    assert captured["model_id"] == "dummy/model"


def test_resolve_backend_reuses_loaded_hf_model(monkeypatch):
    # When an already-loaded model+tokenizer is passed, HF backend should reuse them.
    backend = resolve_backend(
        "hf", model_id="dummy/model", device="cpu",
        hf_model=object(), hf_tokenizer=object(),
    )
    assert backend.name == "hf"
