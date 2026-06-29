"""
Test fixtures using Qwen/Qwen1.5-0.5B-Chat — a real multilingual model that is
already cached locally and small enough for fast CPU-only tests.
"""

import pytest

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
def prompt_pairs():
    """Semantically equivalent (English, non-English) reasoning prompt pairs."""
    return [
        (
            "Solve step by step: what is 15 percent of 80?",
            "ทีละขั้นตอน: 80 ร้อยละ 15 คือเท่าไร?",
        ),
        (
            "What is the capital city of France?",
            "Quelle est la capitale de la France?",
        ),
        (
            "Explain why the sky appears blue during the day.",
            "กรุณาอธิบายว่าทำไมท้องฟ้าจึงดูเป็นสีฟ้าในเวลากลางวัน",
        ),
        (
            "List the steps to solve a quadratic equation.",
            "Bitte liste die Schritte zur Lösung einer quadratischen Gleichung auf.",
        ),
    ]


@pytest.fixture
def forcing_pairs():
    """Language-forcing prompt pairs (English-forcing, target-language-forcing)."""
    return [
        (
            "Please respond in English. The answer is:",
            "กรุณาตอบเป็นภาษาไทย คำตอบคือ:",
        ),
        (
            "Answer in English only:",
            "Répondez uniquement en français:",
        ),
        (
            "Continue in English:",
            "Bitte antworten Sie auf Deutsch:",
        ),
        (
            "Your response must be in English. Here is the result:",
            "คำตอบของคุณต้องเป็นภาษาไทย นี่คือผลลัพธ์:",
        ),
    ]
