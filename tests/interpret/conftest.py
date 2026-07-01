"""Fixtures for interpret tests."""

import pytest

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

MODEL_ID = "Qwen/Qwen1.5-0.5B-Chat"


@pytest.fixture(scope="session")
def model():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not installed")
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    m.eval()
    return m


@pytest.fixture(scope="session")
def tokenizer():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not installed")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def en_texts():
    return [
        "Solve step by step: what is 15 percent of 80?",
        "What is the capital of France?",
    ]


@pytest.fixture
def tgt_texts():
    return [
        "ทีละขั้นตอน: 80 ร้อยละ 15 คือเท่าไร?",
        "เมืองหลวงของฝรั่งเศสคืออะไร?",
    ]
