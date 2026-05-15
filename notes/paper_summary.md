# LatentMAS Paper Summary

Paper: Latent Collaboration in Multi-Agent Systems
arXiv: https://arxiv.org/abs/2511.20639
Upstream code: https://github.com/Gen-Verse/LatentMAS

## Working Summary

LatentMAS moves collaboration among LLM agents from explicit token-space messages into latent-space working memory. The intended benefits are lower token usage, faster wall-clock runtime, and competitive or better reasoning accuracy compared with standard text-based multi-agent systems.

## Reproduction Focus

This workspace starts with a local smoke test rather than full paper reproduction:

- Baseline single-agent GSM8K run
- TextMAS sequential GSM8K run
- LatentMAS sequential GSM8K run
- `Qwen/Qwen3-4B` rather than the heavier `Qwen/Qwen3-14B`
- Hugging Face backend first; vLLM deferred

## Full Reproduction Later

A full run should expand to the paper tasks: GSM8K, AIME 2024/2025, GPQA, ARC Easy/Challenge, MBPP+, HumanEval+, and MedQA, with Qwen3-14B and the same method/prompt settings reported by the paper.
