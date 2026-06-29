#!/usr/bin/env python3
"""Calibrate UniversalLatentSpace adapters on parallel MGSM hidden states.

This trains only the lightweight latent_coordination adapters. The base LLM is
kept frozen and is used only to extract hidden states for semantically matched
MGSM questions across languages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data import load_mgsm  # noqa: E402
from latent_coordination.agents.base_agent import AgentConfig  # noqa: E402
from latent_coordination.agents.specialized_agents import ReasoningAgent  # noqa: E402
from latent_coordination.latent_space.universal_space import UniversalLatentSpace  # noqa: E402


def parse_languages(value: str) -> List[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def iter_mgsm_questions(languages: List[str], max_examples: int) -> Dict[str, List[str]]:
    questions: Dict[str, List[str]] = {}
    for lang in languages:
        rows = []
        for idx, item in enumerate(load_mgsm(split="test", lang=lang)):
            if idx >= max_examples:
                break
            rows.append(str(item["question"]))
        questions[lang] = rows
    return questions


def make_extractor(args: argparse.Namespace) -> ReasoningAgent:
    cfg = AgentConfig(
        agent_id="mgsm_state_extractor",
        model_id=args.model_name,
        role="reasoning",
        device=args.device,
        max_new_tokens=1,
        hidden_dim=args.hidden_dim,
        dtype=args.dtype,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    return ReasoningAgent(cfg)


def pool_hidden(hidden: torch.Tensor, strategy: str) -> torch.Tensor:
    hidden = hidden.detach().float().cpu()
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden state with shape (B, L, D), got {tuple(hidden.shape)}")
    if strategy == "last_token":
        return hidden[:, -1, :]
    if strategy == "mean":
        return hidden.mean(dim=1)
    raise ValueError(f"Unknown pool strategy: {strategy}")


def collect_states(args: argparse.Namespace, languages: List[str]) -> Tuple[torch.Tensor, List[Dict]]:
    questions = iter_mgsm_questions(languages, args.train_examples)
    extractor = make_extractor(args)
    vectors: List[torch.Tensor] = []
    rows: List[Dict] = []

    for lang in languages:
        for idx, question in enumerate(questions[lang]):
            print(f"[collect] {lang} idx={idx}", flush=True)
            states = extractor.extract_hidden_states(question, layer_ids=[args.layer])
            pooled = pool_hidden(states[args.layer], args.pool).squeeze(0)
            vectors.append(pooled)
            rows.append({"lang": lang, "idx": idx, "question": question})

    if not vectors:
        raise RuntimeError("No calibration states were collected.")
    return torch.stack(vectors, dim=0), rows


def build_example_index(rows: List[Dict]) -> Dict[int, List[int]]:
    by_idx: Dict[int, List[int]] = {}
    for row_i, row in enumerate(rows):
        by_idx.setdefault(int(row["idx"]), []).append(row_i)
    return by_idx


def train_uls(args: argparse.Namespace, states: torch.Tensor, rows: List[Dict]) -> Dict:
    dev = torch.device(args.train_device or args.device)
    uls = UniversalLatentSpace(
        universal_dim=args.universal_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        dropout_rate=args.dropout,
        device=str(dev),
    )
    agent_ids = ["mgsm_translator", "mgsm_reasoner"]
    for agent_id in agent_ids:
        uls.register_agent(agent_id, hidden_dim=args.hidden_dim)

    for entry in uls._agents.values():  # local calibration utility; keep adapters trainable.
        entry.encoder.train()
        entry.decoder.train()

    params = []
    for entry in uls._agents.values():  # UniversalLatentSpace is a registry, not nn.Module.
        params.extend(entry.encoder.parameters())
        params.extend(entry.decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    x_all = states.to(dev)
    by_idx = build_example_index(rows)
    losses: List[Dict] = []

    for epoch in range(args.epochs):
        perm = torch.randperm(x_all.shape[0], device=dev)
        epoch_total = 0.0
        epoch_rec = 0.0
        epoch_role = 0.0
        steps = 0

        for start in range(0, x_all.shape[0], args.batch_size):
            batch_idx = perm[start : start + args.batch_size]
            x = x_all[batch_idx]
            opt.zero_grad()

            z_t = uls.encode("mgsm_translator", x)
            z_r = uls.encode("mgsm_reasoner", x)
            rec_t = uls.decode("mgsm_translator", z_t)
            rec_r = uls.decode("mgsm_reasoner", z_r)

            rec_loss = F.mse_loss(rec_t, x) + F.mse_loss(rec_r, x)
            role_loss = (1.0 - F.cosine_similarity(z_t, z_r, dim=-1)).mean()
            loss = rec_loss + args.role_alignment_weight * role_loss

            if args.language_alignment_weight > 0:
                # Align semantically equivalent same-index MGSM problems in hub space.
                lang_losses = []
                batch_rows = set(int(i) for i in batch_idx.detach().cpu().tolist())
                for problem_idx, row_ids in by_idx.items():
                    selected = [i for i in row_ids if i in batch_rows]
                    if len(selected) < 2:
                        continue
                    group_x = x_all[selected]
                    group_z = torch.cat(
                        [
                            uls.encode("mgsm_translator", group_x),
                            uls.encode("mgsm_reasoner", group_x),
                        ],
                        dim=0,
                    )
                    target = group_z.mean(dim=0, keepdim=True)
                    lang_losses.append(1.0 - F.cosine_similarity(group_z, target, dim=-1).mean())
                if lang_losses:
                    lang_loss = torch.stack(lang_losses).mean()
                    loss = loss + args.language_alignment_weight * lang_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()

            epoch_total += float(loss.item())
            epoch_rec += float(rec_loss.item())
            epoch_role += float(role_loss.item())
            steps += 1

        row = {
            "epoch": epoch + 1,
            "loss": epoch_total / max(steps, 1),
            "reconstruction_loss": epoch_rec / max(steps, 1),
            "role_alignment_loss": epoch_role / max(steps, 1),
        }
        losses.append(row)
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                "[train] epoch={epoch} loss={loss:.6f} rec={reconstruction_loss:.6f} "
                "role={role_alignment_loss:.6f}".format(**row),
                flush=True,
            )

    for entry in uls._agents.values():
        entry.encoder.eval()
        entry.decoder.eval()

    with torch.no_grad():
        metrics = {
            agent_id: uls.compute_transfer_quality(agent_id, x_all)
            for agent_id in agent_ids
        }
        transfer = uls.transfer("mgsm_translator", "mgsm_reasoner", x_all, record_transfer=False)
        metrics["translator_to_reasoner"] = {
            "cosine_to_original": float(F.cosine_similarity(transfer, x_all, dim=-1).mean().item()),
            "mse_to_original": float(F.mse_loss(transfer, x_all).item()),
        }

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    uls.save_adapters(str(out_dir))
    meta = {
        "model_name": args.model_name,
        "languages": args.languages,
        "train_examples": args.train_examples,
        "layer": args.layer,
        "pool": args.pool,
        "hidden_dim": args.hidden_dim,
        "universal_dim": args.universal_dim,
        "adapter_hidden_dim": args.adapter_hidden_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "role_alignment_weight": args.role_alignment_weight,
        "language_alignment_weight": args.language_alignment_weight,
        "metrics": metrics,
        "losses": losses,
    }
    (out_dir / "calibration_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] saved calibrated ULS adapters to {out_dir}", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--languages", default="bn,de,en,es,fr,ja,ru,sw,te,th,zh")
    parser.add_argument("--train_examples", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--hidden_dim", type=int, default=2560)
    parser.add_argument("--universal_dim", type=int, default=256)
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pool", choices=["last_token", "mean"], default="last_token")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--role_alignment_weight", type=float, default=0.1)
    parser.add_argument("--language_alignment_weight", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--output",
        default="results/latent_coordination_mgsm_plain/Qwen3-4B/mgsm_uls_adapters_first50",
    )
    args = parser.parse_args()

    languages = parse_languages(args.languages)
    states, rows = collect_states(args, languages)
    train_uls(args, states, rows)


if __name__ == "__main__":
    main()
