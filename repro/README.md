# Reproducing LatentMAS on UCI HPC3

## What the code actually does (mechanism)

`run.py` → `ModelWrapper` (`models.py`) → one of three methods in `methods/`.
All three run the same 4 agents: **Planner → Critic → Refiner → Judger**
(`methods/__init__.py:default_agents`). Only the handoff differs.

- **`baseline`** — one prompt, one generation. No agents.
- **`text_mas`** — each agent generates *text*; the text is appended to `contexts[idx]`
  and pasted into the next agent's prompt. This is the token-space baseline.
- **`latent_mas`** — the interesting one. Planner/Critic/Refiner **never generate text**.

### The LatentMAS loop (`models.py:generate_latent_batch`)

For each non-judger agent:
1. Its prompt is forward-passed *on top of the running `past_key_values`*, so the
   KV cache from previous agents is the entire inter-agent channel.
2. Then `latent_steps` autoregressive steps happen **in embedding space**: the last
   layer's final hidden state is fed straight back in as `inputs_embeds`, no
   sampling, no detokenization. Each step appends one entry to the KV cache.
3. The judger is the only agent that calls `generate()`, and it does so with
   `past_key_values=past_kv` — inheriting every previous agent's KV.

`--latent_space_realign` (`models.py:_build_latent_realign_matrix`) is the
"training-free alignment": it least-squares-solves `W_out @ M ≈ W_in` so a hidden
state (output space) is mapped back into input-embedding space, then rescales it to
the mean input-embedding norm. Without the flag `M` is set to identity but the norm
rescaling still happens.

The efficiency claim falls out of this directly: 3 of 4 agents emit zero tokens.

---

## Findings that affect reproduction

### 0. `~/.local` leaks into every python3.10 env on this account (READ FIRST)

`~/.local/lib/python3.10/site-packages` is 931 MB and `site.ENABLE_USER_SITE` is
True, so that directory is on `sys.path` of **every** python3.10 environment you
create, conda envs included:

```
$ /pub/$USER/envs/latentmas/bin/python -c "import site,sys; print(site.ENABLE_USER_SITE, [p for p in sys.path if '.local' in p])"
True ['/data/homezvol1/ynagaraj/.local/lib/python3.10/site-packages']
```

Two consequences:

1. **It is why the original `latentmas` env was broken.** `~/.local` holds
   `nvidia-nccl-cu12 2.19.3`, which shadows whatever nccl the env's own torch
   shipped -> `undefined symbol: ncclCommResume`. The mismatch is not *inside*
   the env, so rebuilding the env never fixes it.

2. **A bare `pip install` into a fresh env is destructive.** pip treats the
   `~/.local` packages as "existing installations" and uninstalls them out of your
   home directory. On 2026-08-26 an unguarded run of `00_setup_env.sh` removed 18
   packages from `~/.local` (click, fsspec, hf-xet, safetensors, setuptools,
   tokenizers, and all 12 `nvidia-*-cu12`), breaking `fastai`, `synthcity`,
   `pytorch-lightning`, `be-great` and friends until they were restored at their
   exact prior versions.

**Fix, applied to every script in this directory:**

```bash
export PYTHONNOUSERSITE=1     # hides ~/.local from the interpreter AND from pip
export PIP_USER=0
```

`00_setup_env.sh` now asserts no `.local` path reaches `sys.path` before it
declares the env ready. Keep that assertion.

### 1. `--latent_steps` defaults to 0, which disables the latent channel entirely
`methods/latent_mas.py`:
```python
past_for_decoding = past_kv if self.latent_steps > 0 else None
```
With `latent_steps=0` the judger gets **`None`** — the whole accumulated KV cache is
thrown away, and `latent_mas` degenerates to a single-agent judger call.
**Always pass `--latent_steps` explicitly.** The README says tune it over [0, 80].

### 2. The two released reference logs disagree on `latent_steps`
| log | latent_steps | reported |
|---|---|---|
| `example_logs/qwen3_14b_mbppplus_sequential.txt` | **0** (all 1134 entries) | 76.19% (288/378) |
| `example_logs/qwen3_14b_humanevalplus_hierarchical.txt` | **10** (all 492 entries) | — |

So the released MBPP+ log was produced with the latent channel switched off per (1).
Worth confirming against the paper's Table 1 MBPP+/14B cell before you treat 76.19%
as the LatentMAS number. (The log's trailing JSON also still says `"method": "muscle"`,
an older internal name.)

### 2b. The released log headers disagree with the README commands (IMPORTANT)

Both reference logs print their full arg namespace as the first lines. Those are
the settings that actually produced the published numbers, and they are **not**
what the README's example commands give you:

| arg | README example | `mbppplus` log | `humanevalplus` log |
|---|---|---|---|
| `--max_new_tokens` | 2048 | **4096** | **4096** |
| `--think` | never mentioned | **True** | **True** |
| `--generate_bs` | 20 (default) | 15 | 20 |
| `--latent_steps` | "tune over [0,80]" | 0 | 10 |
| `--temperature` / `--top_p` | 0.6 / 0.95 | 0.6 / 0.95 | 0.6 / 0.95 |
| `--seed` | 42 | 42 | 42 |

`--think` appends a literal `<think>` to the rendered assistant turn
(`methods/latent_mas.py:112`), forcing Qwen3 into thinking mode. It is only wired
into `latent_mas` -- `baseline` and `text_mas` accept the flag and ignore it, relying
on Qwen3's chat template defaulting `enable_thinking=True` instead. `models.py:render_chat`
never passes `enable_thinking`, so that default stands.

`run_latentmas.sbatch` now defaults `MAXNEW=4096`, `BS=15`, `THINK=1` to match.

### 2c. `run.py` reports no token counts

The final JSON line carries `accuracy`, `correct`, `total_time_sec`,
`time_per_sample_sec` -- and nothing about tokens. The paper's central efficiency
claim is a token claim, so it cannot be checked from run.py output alone.

`repro/analyze.py` closes this: it parses the stdout log, re-tokenizes every
agent's `[Output]` block with the model's own tokenizer, and reports generated
tokens per problem broken down by agent role, plus token/speed deltas between
methods. Validated against `example_logs/qwen3_14b_mbppplus_sequential.txt`:
parses all 378 problems and reproduces the reported 0.7619 exactly.

### 3. `methods/latent_mas.py` imports vLLM unconditionally — hard crash
`from vllm import SamplingParams` at module top, but `requirements.txt` has no vllm
and the README says *"Use the HF backend to reproduce the official published results."*
Fixed by `02_patch_blockers.sh`.

### 4. `run.py --model_name` choices omit Qwen3-8B
Lists `Qwen/Qwen3-4B` twice. The 8B command in the README fails argparse. Also patched.

### 5. Batched generation right-pads (quality risk)
Qwen3's `tokenizer_config.json` sets no `padding_side`, so HF defaults to **right**,
and nothing in `models.py` overrides it. With `--generate_bs 20` every sequence
shorter than the batch max continues from a `<|endoftext|>` pad token rather than its
real last prompt token. Decoding still looks fine (`skip_special_tokens=True` strips
the pads) so this fails silently.

Do **not** blindly set `padding_side="left"`: `generate_text_batch` slices
`sequences[idx, prompt_len:]`, which is only correct under right padding. A left-pad
fix also needs that slice changed to `sequences[idx, input_ids.shape[1]:]`.

**Recommended:** run a 30-sample subset at `--generate_bs 1` and at `--generate_bs 20`
and compare. That quantifies the artifact before you commit GPU-hours.

### 6. Results are nondeterministic
`do_sample=True, temperature=0.6, top_p=0.95`. `--seed` is set, but batching changes
RNG consumption. Expect ±1–2% run-to-run on 378 samples; more on AIME (n=30).

### 7. Nothing is written to disk
Accuracy only goes to stdout as a final JSON line. `run_latentmas.sbatch` tees it.
Logs are large — the released 378-problem log is 8.7 MB because full prompts are printed.

---

## Compute plan for this cluster

Your account is `ynagaraj`; QOS `normal/high/low`. Limits: `free-gpu` 24 GPUs,
`free-gpu32` **4 GPUs**. `free-*` partitions are preemptible — use `gpu`/`gpu32` (billed)
for anything long.

`models.py` hardcodes `torch.bfloat16`. **V100 is sm_70 and has no bf16** — never
schedule there, even though it is the most idle pool.

| model | bf16 weights | where |
|---|---|---|
| Qwen3-4B | ~8 GB | `free-gpu` + `gpu:A30:1` (24 GB) |
| Qwen3-8B | ~16 GB | `free-gpu` + `gpu:A30:1` (tight w/ KV) or L40S |
| Qwen3-14B | ~29 GB | `free-gpu32` + `gpu:L40S:1` or `gpu:RTX6000:1` (48 GB), or `free-gpu` + `gpu:A100:1` |

Storage: `$HOME` is 50 GB with 25 GB free — too small. Put the env **and** `HF_HOME`
under `/pub/$USER` (beegfs, 513 TB free). The three Qwen3 checkpoints total ~55 GB.

Dataset sizes: gsm8k 1319 · mbppplus 378 · humanevalplus 164 · arc_easy 2376 ·
arc_challenge 1172 · gpqa 198 · medqa 300 · aime24/25 30 each.

---

## Order of operations

```bash
cd ~/LatentMAS-Explore

./repro/00_setup_env.sh          # login node, ~10 min. Do NOT reuse the existing
                                 # `latentmas` env (torch 2.13 + nccl-cu12 2.19.3
                                 # mismatch -> undefined symbol: ncclCommResume)
./repro/01_fetch_assets.sh       # login node, ~55 GB into /pub/$USER/hf
./repro/02_patch_blockers.sh     # fixes #3 and #4

# Smoke test: 8 samples, 4B, cheap A30. Confirms the whole path end to end.
sbatch --partition=free-gpu --gres=gpu:A30:1 --time=1:00:00 \
  --export=ALL,MODEL=Qwen/Qwen3-4B,TASK=gsm8k,METHOD=latent_mas,PROMPT=sequential,LSTEPS=10,NSAMP=8,BS=4 \
  repro/run_latentmas.sbatch

# Padding probe (finding #5): same 30 samples, bs=1 vs bs=20.
# Then the two runs that have released reference logs to diff against:
sbatch --export=ALL,MODEL=Qwen/Qwen3-14B,TASK=mbppplus,METHOD=latent_mas,PROMPT=sequential,LSTEPS=0,NSAMP=-1,BS=20 \
  repro/run_latentmas.sbatch     # target 76.19%
sbatch --export=ALL,MODEL=Qwen/Qwen3-14B,TASK=humanevalplus,METHOD=latent_mas,PROMPT=hierarchical,LSTEPS=10,NSAMP=-1,BS=20 \
  repro/run_latentmas.sbatch

# The claim is comparative, so each LatentMAS run needs its two partners:
#   METHOD=baseline  (no --prompt effect)
#   METHOD=text_mas  (same --prompt) -- ~4x the wall clock, all 4 agents generate
```

Rough wall clock, Qwen3-14B on one L40S, 378 problems @ 2048 max_new_tokens:
LatentMAS ~1–3 h, baseline similar, TextMAS ~4–8 h. Budget accordingly; the
`--time=12:00:00` default in the sbatch file is sized for TextMAS.

### 8. Dependency drift breaks the repo on a 2026 pip (both fixed)

`requirements.txt` pins nothing (`transformers`, `torch`, `numpy`, `tqdm`,
`accelerate`, `datasets`), so a fresh install in 2026 resolves to versions the code
predates.

**8a. transformers 5.x removes the legacy Cache API.** A bare install gives
transformers 5.16.1, where `DynamicCache.to_legacy_cache` / `.from_legacy_cache`
no longer exist:

```
>>> hasattr(DynamicCache, "to_legacy_cache")
False
```

`methods/latent_mas.py:68,73` (`_truncate_past`) calls both. In practice this is
*latent*, not fatal: `_truncate_past` only runs when `latent_only` or
`sequential_info_only` is set, and both come from `getattr(args, ..., False)` while
`run.py` defines neither -- so they are always False. It would fire the moment
anyone adds those flags. transformers 5.x also deprecates `torch_dtype=`
(`models.py:59,76`).

`00_setup_env.sh` now pins `transformers==4.57.1`, the version the pre-existing
env carried and the paper-era API surface.

**8b. `data.py` used the legacy bare dataset name `gsm8k`.** Modern
`huggingface_hub` requires `namespace/name` and raises:

```
HfUriError: Invalid HF URI 'hf://datasets/gsm8k@.../.huggingface.yaml'.
Repository id must be 'namespace/name', got 'gsm8k'.
```

Fixed in `data.py:9` -> `openai/gsm8k` (same dataset, canonical id). It was the only
bare name; the other nine loaders are already namespaced.

### 9. LatentMAS costs PEAK MEMORY, and the efficiency framing omits it

Measured on GSM8K / Qwen3-4B / `--max_new_tokens 4096`, one 24 GB A30, `bs=15`:

| arm | outcome |
|---|---|
| `latent_mas` (ls=10) | **CUDA OOM** |
| `baseline` | ran fine |
| `text_mas` | ran fine |

Same GPU, same batch size, same data. The OOM lands in
`transformers/integrations/sdpa_attention.py:27, in repeat_kv`:

```
Tried to allocate 582.00 MiB. GPU 0 has a total capacity of 23.60 GiB of which
561.44 MiB is free. Of the allocated memory 18.39 GiB is allocated by PyTorch,
and 4.33 GiB is reserved by PyTorch but unallocated.
```

The mechanism explains it: the judger generates with
`past_key_values=past_kv`, where `past_kv` is the **concatenated** KV of the
planner + critic + refiner (each contributing its full prompt plus `latent_steps`
entries). So the judger attends over a context several times longer than any single
agent's prompt, and it does so for all `max_new_tokens` decode steps.

That is the same property the token savings come from -- 3 agents emit no text
*because* their contribution lives in the KV cache -- so the memory cost is
intrinsic, not an implementation wart. It just is not part of the paper's
"efficiency" story, which reports tokens and wall-clock only.

**Practical consequences:**
- `bs=15` (the released mbppplus setting) needs a 48 GB card for a 4B model at
  4096 new tokens. The paper ran 14B, so their hardware was larger still.
- Peak memory scales with `batch x (sum of all agent prompt lengths + latent_steps
  x n_agents + max_new_tokens)`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` recovers the 4.33 GiB of
  reserved-but-unallocated fragmentation; now set in `run_latentmas.sbatch`.

**Benchmarking note:** wall-clock speedup is half the claim, so every arm must run
on the SAME GPU model. Do not move only the OOM-ing arm to a bigger card. All arms
were moved to L40S together.


---

## 10. The two released logs ARE two paper cells — confirmed to the token

`repro/analyze.py` re-tokenizes every agent's `[Output]` block. Run against the two
released logs with the Qwen3-14B tokenizer, it reproduces the paper's **Token**
column exactly:

| released log | acc (log / paper) | gen tok/prob (log / paper) | paper cell |
|---|---|---|---|
| `qwen3_14b_mbppplus_sequential.txt` | 76.19 / 75.7 | **1621.2 / 1621** | Table 1, 14B, MBPP+, sequential |
| `qwen3_14b_humanevalplus_hierarchical.txt` | 84.76 / 86.6 | **1512.3 / 1512** | Table 2, 14B, HumanEval+, hierarchical |

An exact match on both token figures is not a coincidence: these logs are the runs
behind those cells. That pins down two things the paper does not state.

### 10a. The paper's MBPP+/14B LatentMAS number was produced with the latent channel OFF

Per finding #2 the mbppplus log has `latent_steps: 0` in all 1134 trace entries.
Combined with #1, the judger receives `past_key_values=None`. Verified live with
`repro/trace_pipeline.py` on Qwen3-4B/GSM8K:

```
latent_steps=10                          latent_steps=0
  Planner  KV   0 ->  166                  Planner  KV   0 ->  156
  Critic   KV 166 ->  373                  Critic   KV 156 ->  353
  Refiner  KV 373 ->  566                  Refiner  KV 353 ->  536
  Judger   inherits 566  <-- channel       Judger   inherits   0  <-- DISCARDED
```

So the published 75.7 on MBPP+/14B/sequential is a single-agent judger call preceded
by three forward passes whose entire output is thrown away. Note the released logs
print `method: muscle` and their arg namespace contains no `latent_steps` field at
all, so they came from an *earlier code revision*; today's code at ls=0 is the
degenerate path, whatever the old code did.

### 10b. "Token" counts generated text only — prefill is not in it

`analyze.py` now also counts the `[To Tokenize]` blocks:

| released log | generated/prob | prompt/prob | sum |
|---|---|---|---|
| mbppplus (ls=0) | 1621.2 | **785.0** | 2406.1 |
| humanevalplus (ls=10) | 1512.3 | **1182.9** | 2695.2 |

LatentMAS forward-passes **four** agent prompts to build the KV cache. A single
agent forward-passes one. That ~4x prefill is real work and appears nowhere in the
Token column, which is why the efficiency framing reads stronger than the compute
picture.

**State this carefully.** Prefill is parallel and far cheaper per token than
sequential decode, so `generated + prompt` is *not* an apples-to-apples cost metric
either. The honest reading: the paper measures decode tokens, which is the dominant
cost and a legitimate metric; but the omitted prefill is what shows up in the two
places the savings do not materialize — wall-clock (finding #11) and peak memory
(finding #9).

## 11. Wall-clock: LatentMAS is slower than the single-agent baseline here

Measured on one L40S, Qwen3-4B, MBPP+, bs=15, max_new_tokens=4096, think=1:

| arm | sec/problem (running ETA) |
|---|---|
| `baseline` | ~27 |
| `latent_mas` ls=10 | ~40 |

That is ~1.7x *slower*. The paper's own Table 1 has LatentMAS slower than Single too
(4B MBPP+: Speed 577 vs 523, ~1.1x), so the direction agrees — its headline **x3.7
speedup is against TextMAS, not against a single agent**. Any summary that says
"LatentMAS is faster" without naming the comparison arm is wrong.

## 12. HF itself warns about the right-padding artifact (finding #5 confirmed)

Both arms emit, unprompted, from `transformers`:

```
A decoder-only architecture is being used, but right-padding was detected!
For correct generation results, please set `padding_side='left'`.
```

This fires on every batch in every arm, including the released-log configuration.
Keep right-padding for reproduction fidelity — it is what the authors ran — but the
artifact is real and applies to the published numbers too. Quantify it with the
bs=1 vs bs=15 probe rather than silently "fixing" it.

---

## 13. Billed GPU partitions are NOT available to this account — preemption is unavoidable

`sacctmgr show assoc user=ynagaraj` lists exactly one account, `ynagaraj`
(998 of 1000 SUs free). That account is not authorized for the billed GPU
partitions, and Slurm rejects submission at the job_submit plugin:

```
$ sbatch --partition=gpu32 --gres=gpu:L40S:1 ...
sbatch: error: !!! slurm_job_submit ERROR: submitted a job to a gpu32 partition,
        without specifying a GPU32 Account.
$ sbatch --partition=gpu --gres=gpu:A100:1 ...
sbatch: error: !!! slurm_job_submit ERROR: submitted a job to a gpu partition,
        without specifying a GPU Account.
```

`free-gpu` / `free-gpu32` accept the account and schedule immediately. So the
advice in the compute plan above to "swap to gpu / gpu32 (billed) for long runs"
is **not actionable here** — every run is preemptible, and a 378-problem arm takes
3–4 h. Getting a GPU allocation account requires a PI request to HPC3 support.

### 13a. Consequence: checkpoint/resume, not a better partition

Preemption destroyed two arms that were 52% and 32% done (jobs 55644121 /
55644122, 2.5 GPU-hours) because `run.py` persisted nothing until its final line.

`run.py` now takes `--checkpoint PATH`:

- appends one JSON record per completed problem to a JSONL file **after every
  batch** (the batch is the only boundary the model actually completes on),
  `flush()` + `fsync()` so a SIGKILL cannot lose a completed batch;
- on startup replays the file, seeds `preds`/`processed` from it, and skips that
  many dataset items. Loaders are deterministic and unshuffled, so the completed
  set is exactly the first N items and resume is a prefix skip;
- a truncated final line from a hard kill stops the replay instead of raising;
- carries `_elapsed_sec` forward, so `total_time_sec` spans restarts rather than
  measuring only the last attempt;
- reports `resumed_from` in the final JSON, so a stitched run is never mistaken
  for a single pass.

`run_latentmas.sbatch` sets `#SBATCH --requeue` + `--open-mode=append` and passes
`--checkpoint repro/results/${TAG}.ckpt.jsonl`. The path is keyed on the **config**,
not the job id, so a requeued attempt (same job id) and a hand-resubmitted job
both pick the work back up. The stdout log is `tee -a`'d for the same reason.

Verified end to end on GSM8K/Qwen3-4B before launching:

| pass | behaviour |
|---|---|
| 1 (`--max_samples 4`) | wrote 4 records, `resumed_from: 0`, `total_time_sec 18.8` |
| 2 (`--max_samples 8`) | `[resume] restored 4/8`, continued at Problem #5, `resumed_from: 4`, `total_time_sec 35.7` (= 18.8 + 16.9) |
| 3 (rerun of pass 2) | `restored 8/8`, no duplicate records, no recomputation |

The original `run.py` is preserved at `repro/run.py.orig`.

### 13b. Submitting the three-arm comparison

`repro/submit_mbppplus_4b.sh` — Qwen3-4B / MBPP+ / bs=15 / think=1 /
max_new_tokens=4096, all three arms pinned to `gpu:L40S:1` on `free-gpu32` so
wall-clock stays comparable (finding #11):

| arm | why |
|---|---|
| `baseline` | single agent; the cost floor LatentMAS is measured against |
| `latent_mas` `ls=0` | the released-log config; degenerate per findings #1/#10a |
| `latent_mas` `ls=10` | the latent channel actually switched on |

**Still missing: `text_mas`.** The paper's headline x3.7 speedup is measured
against TextMAS, not against a single agent (finding #11), so the headline claim
cannot be checked until that arm runs. It is the most expensive arm — all four
agents generate text — so budget roughly 4x the baseline wall-clock.

---

## Session 2026-08-28 (afternoon): the four-arm MBPP+/Qwen3-4B comparison

### 9. Preemption here is CANCEL, not requeue — `--requeue` is dead weight

Four jobs died on 2026-08-28 with

```
error: *** JOB 55645526 ON hpc3-gpu-k54-07 CANCELLED AT 2026-08-28T14:55:57 DUE to SIGNAL Terminated ***
```

and `sacct` shows them `CANCELLED`, never `REQUEUED` — `#SBATCH --requeue` did not
fire once. The reason is visible in `sinfo`: the **free-gpu32 L40S nodes
`hpc3-gpu-k54-06/07/08` are the same physical nodes as the billed `gpu32`
partition.** A paying gpu32 job evicts us, and the cluster's preemption mode is
cancel. `free-gpu32`'s other nodes (`m54-*`, `n54-*`) are RTX6000 — Turing sm_75,
**no bf16**, and `models.py` hardcodes bf16 — so they are not an escape hatch.

**Fix:** `run_latentmas.sbatch` now resubmits *itself*. It traps SIGTERM and also
checks the exit status, and in either case re-`sbatch`es with `CHAIN_N` incremented
(`CHAIN_MAX=20`). Combined with `--checkpoint` each attempt resumes where the last
stopped. A guard blocks the chain when the checkpoint did not grow during the
attempt, so a genuine crash (bad flag, OOM) fails once instead of twenty times.
Knobs: `CHAIN=0` disables, `GPUMODEL`/`JOBMEM`/`JOBTIME` are carried to the child so
the GPU model stays pinned across the whole chain.

### 10. `repro/compare_arms.py` — compare arms *while they are still running*

`analyze.py` parses a finished stdout log. Under constant preemption a finished log
is the exception, so `compare_arms.py` reads the `--checkpoint` JSONL instead. Each
record already carries the full agent trace, so everything the paper compares on is
recoverable mid-run:

- `agents[i].input_ids` → prompt/prefill tokens per agent role
- `agents[i].output` → re-tokenized for generated tokens per agent role
- `agents[i].latent_steps` → KV entries written by the latent channel
- `_elapsed_sec` → wall-clock at each batch boundary, carried across preemption

Everything is reported on the **common prefix** all arms have finished (resume is a
prefix skip, so record *i* is problem *i* in every arm), and the timing for a prefix
of length *n* is read at the batch boundary ≤ *n* — so a 150-problem arm is compared
against a 15-problem arm on the first 15 problems only, not on its own average.

### 11. First live numbers — the mechanism is real, the accounting is three-way

MBPP+, Qwen3-4B, sequential, bs=15, think=1, max_new_tokens=4096, all on **L40S**.
Common prefix of only **15 problems** — accuracy here is noise, the token and time
columns are not:

| arm | acc | sec/prob | gen tok | prompt tok | latent steps |
|---|---|---|---|---|---|
| baseline | 80.0% | 17.9 | 998.9 | 225.0 | 0 |
| latent_mas ls=0 | 73.3% | 30.0 | 1355.6 | 1021.0 | 0 |
| latent_mas ls=10 | 40.0% | 40.7 | **702.1** | **1021.0** | 30 |

**The latent channel does exactly what the paper says it does.** Per-role generated
tokens at ls=10 are `judger=702.1` and nothing else: planner, critic and refiner
produce zero characters of text. That is the mechanism, confirmed end to end.

But the cost is three currencies, and the paper's Token column reports one:

1. **Generated tokens fall** (999 → 702 vs the single-agent baseline).
2. **Prompt tokens rise 4.5x** (225 → 1021), because four agent prompts are
   forward-passed to build the KV cache. Finding #2c already established the paper
   omits prefill; this is what it omits.
3. **30 latent steps** per problem are neither generated tokens nor prompt tokens —
   a third currency the Token column has no column for.

And **wall-clock moves the wrong way: 2.3x slower than baseline**, consistent with
the earlier GSM8K measurement (~1.7x slower). The paper's 3.7x speedup is measured
against TextMAS, not against a single agent — which is why the `text_mas` arm is the
one that actually tests the claim.

### 12. Current run status

Four arms, all MBPP+/Qwen3-4B/L40S, all checkpointed and chained:

| arm | where | note |
|---|---|---|
| `baseline` | job 55646145 | resumed from 135/378 |
| `latent_mas` ls=10 | job 55646144 | the method, actually switched on |
| `latent_mas` ls=0 | job 55646146 | the released-log config; queued |
| `text_mas` | interactive job 55646079 | ~60 s/problem → ~6 h for 378 |

`text_mas` is running in the *interactive* job, which has no chain. If that job ends,
relaunch it as a batch job — the checkpoint makes it resume:

```bash
sbatch --partition=free-gpu32 --gres=gpu:L40S:1 --mem=32G --time=12:00:00 \
  --export=ALL,METHOD=text_mas,MODEL=Qwen/Qwen3-4B,TASK=mbppplus,PROMPT=sequential,\
LSTEPS=0,NSAMP=-1,BS=15 repro/run_latentmas.sbatch
```

Never run two processes against one checkpoint file — they both append.

### 13. **The latent thought is seeded from a PAD token in ~94% of sequences**

This is finding #5 (right padding) again, but its consequence for `latent_mas` is far
worse than "a quality risk", and it is the leading explanation for the accuracy
collapse in #11.

`models.py:generate_latent_batch`:

```python
e_t         = outputs.hidden_states[0][:, -1, :]   # [B, D]
last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, D]   <- seeds the latent loop
```

Position `-1` of a **right-padded** batch is a pad token for every sequence shorter
than the batch max. So the latent thought does not start from the end of the agent's
prompt — it starts from the hidden state of `<|endoftext|>`.

It compounds two lines later:

```python
latent_mask = torch.ones((latent_embed.shape[0], past_len + 1), ...)
```

The mask is **all ones**, so every pad position already in the KV cache is *un-masked*.
The latent steps, and then the judger decoding on top of that cache, attend to pad
tokens as if they were content.

At `--generate_bs 1` neither problem exists, which is why it is invisible in a smoke
test. Measured on our own MBPP+/4B/bs=15 run (`latent_mas ... ls10`, 45 problems):

```
batch 0: max=323 min=173  pad/seq=103.0   seqs ending on a PAD: 14/15
batch 1: max=303 min=175  pad/seq= 85.5   seqs ending on a PAD: 14/15
batch 2: max=412 min=174  pad/seq=183.3   seqs ending on a PAD: 14/15
-> 42/45 = 93.3% of sequences seed the latent loop from a pad position
-> 35.8% of all KV positions are pad, and all of them are un-masked
```

**The released reference log is in the same state.** Re-tokenizing the planner
prompts of `example_logs/qwen3_14b_humanevalplus_hierarchical.txt` (14B, ls=10,
`generate_bs=20` — the run that produced the published 84.76%):

```
planner prompt tokens: min=197 max=550 mean=293
151/160 = 94.4% of sequences would end on a PAD at bs=20
32.7% of KV positions would be pad
```

So the published LatentMAS numbers were produced with the latent channel seeded from
pad tokens in ~94% of cases. That does not make them wrong — it makes them a
measurement of a *different* mechanism than the one the paper describes, and it means
the method's ceiling has not actually been measured. Note the 14B/HumanEval+ run still
reached 84.76% under this, while our 4B/MBPP+ run sits near 38%, so the damage is not
uniform across scale or task.

**Open question, being tested now:** job 55646313 runs MBPP+/4B/ls=10 at
`--generate_bs 1` (no padding at all) over the same first 45 problems. If accuracy
returns to the ~69-75% band of the other arms, padding is the cause and the
`latent_mas` numbers at bs>1 are an artifact.

A fix needs three changes together, and **must not be applied while the comparison
arms are running** — a chained resubmit would silently pick up new code mid-run:
1. seed from the real last non-pad index (`attention_mask.sum(-1) - 1`), not `-1`;
2. carry the true attention mask into `latent_mask` instead of `torch.ones`;
3. left-pad instead, which also requires fixing the `sequences[idx, prompt_len:]`
   slice in `generate_text_batch` (finding #5).
