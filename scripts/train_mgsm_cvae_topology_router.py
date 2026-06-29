#!/usr/bin/env python3
"""Train a small MGSM CVAE topology router from route outcome CSVs.

The router learns a query-conditioned graph over two active roles:
  0. Translation latent agent
  1. Reasoning agent

Graphs are labelled from paired experiment outputs:
  - reasoning_only examples.csv
  - trained/direct latent examples.csv

By default, the latent route is labelled only when it fixes an example that
reasoning_only missed. Otherwise the conservative reasoning-only route is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from latent_coordination.topology.cvae_prior import (  # noqa: E402
    CVAETopologyPrior,
    TopologyDataset,
    TrainingConfig,
    beta_annealing_schedule,
)


ROLE_TRANSLATION = 0
ROLE_REASONING = 1


def stable_tokenize(text: str, lang: str, *, vocab_size: int, max_seq_len: int) -> torch.Tensor:
    tokens = []
    for word in f"lang_{lang} {text}".lower().split()[:max_seq_len]:
        h = (int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % (vocab_size - 1)) + 1
        tokens.append(h)
    while len(tokens) < max_seq_len:
        tokens.append(0)
    return torch.tensor(tokens, dtype=torch.long)


def graph_for_route(route: str) -> torch.Tensor:
    graph = torch.zeros(2, 2, dtype=torch.float32)
    if route == "latent":
        graph[ROLE_TRANSLATION, ROLE_REASONING] = 1.0
    return graph


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_label_rows(args: argparse.Namespace) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    reasoning = pd.read_csv(args.reasoning_csv)
    latent = pd.read_csv(args.latent_csv)

    key_cols = ["lang", "idx"]
    needed = set(key_cols + ["correct", "question"])
    for name, df in [("reasoning", reasoning), ("latent", latent)]:
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")

    merged = reasoning.merge(
        latent,
        on=key_cols,
        suffixes=("_reasoning", "_latent"),
    )
    if merged.empty:
        raise ValueError("No overlapping (lang, idx) rows between reasoning and latent CSVs.")

    graphs: List[torch.Tensor] = []
    queries: List[torch.Tensor] = []
    metas: List[Dict] = []

    label_counts = {"reasoning": 0, "latent": 0}
    for row in merged.to_dict("records"):
        reasoning_ok = as_bool(row["correct_reasoning"])
        latent_ok = as_bool(row["correct_latent"])

        if args.label_policy == "latent_if_fix":
            route = "latent" if latent_ok and not reasoning_ok else "reasoning"
        elif args.label_policy == "latent_if_correct":
            route = "latent" if latent_ok else "reasoning"
        elif args.label_policy == "oracle_accuracy":
            route = "latent" if latent_ok and not reasoning_ok else "reasoning"
        else:
            raise ValueError(f"Unknown label policy: {args.label_policy}")

        question = row.get("question_reasoning") or row.get("question_latent")
        lang = str(row["lang"])
        graphs.append(graph_for_route(route))
        queries.append(stable_tokenize(question, lang, vocab_size=args.vocab_size, max_seq_len=args.max_seq_len))
        label_counts[route] += 1
        metas.append(
            {
                "lang": lang,
                "idx": int(row["idx"]),
                "route": route,
                "reasoning_correct": reasoning_ok,
                "latent_correct": latent_ok,
            }
        )

    print(f"[labels] {label_counts}", flush=True)
    return graphs, queries, metas


def train(args: argparse.Namespace) -> None:
    graphs, queries, metas = load_label_rows(args)
    dataset = TopologyDataset(
        graphs=graphs,
        query_tokens=queries,
        metadata=metas,
        max_n_agents=2,
        max_seq_len=args.max_seq_len,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    cfg = TrainingConfig(
        z_dim=args.z_dim,
        query_dim=args.query_dim,
        graph_hidden_dim=args.graph_hidden_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        max_n_agents=2,
        query_vocab_size=args.vocab_size,
        query_embed_dim=args.query_embed_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_n_layers=args.query_layers,
        lr=args.lr,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        beta_max=args.beta_max,
        warmup_epochs=args.warmup_epochs,
        cycle_length=args.cycle_length,
        grad_clip=args.grad_clip,
        checkpoint_interval=max(args.epochs, 1),
        use_transformer_encoder=not args.use_bilstm,
        free_bits_lambda=args.free_bits_lambda,
        device=args.device,
    )
    model = CVAETopologyPrior(cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(args.epochs):
        model.train()
        beta = beta_annealing_schedule(
            epoch,
            args.epochs,
            beta_max=args.beta_max,
            warmup=args.warmup_epochs,
            cycle_length=args.cycle_length,
        )
        totals = {"loss": 0.0, "bce": 0.0, "kl": 0.0}
        batches = 0
        for batch in loader:
            G = batch["G"].to(args.device)
            Q = batch["Q"].to(args.device)
            optimizer.zero_grad()
            recon_G, mu, logvar = model(G, Q)
            bce = F.binary_cross_entropy(recon_G, G, reduction="mean")
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = bce + beta * kl
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals["loss"] += float(loss.item())
            totals["bce"] += float(bce.item())
            totals["kl"] += float(kl.item())
            batches += 1

        row = {
            "epoch": epoch + 1,
            "beta": beta,
            "loss": totals["loss"] / max(batches, 1),
            "bce": totals["bce"] / max(batches, 1),
            "kl": totals["kl"] / max(batches, 1),
        }
        history.append(row)
        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(
                "[train] epoch={epoch} beta={beta:.3f} loss={loss:.4f} bce={bce:.4f} kl={kl:.4f}".format(**row),
                flush=True,
            )

    model.eval()
    with torch.no_grad():
        Q_all = torch.stack(queries).to(args.device)
        z = torch.zeros(Q_all.shape[0], args.z_dim, device=args.device)
        probs = model.decode(z, Q_all)
        edge_probs = probs[:, ROLE_TRANSLATION, ROLE_REASONING].detach().cpu()
        labels = torch.tensor([1.0 if meta["route"] == "latent" else 0.0 for meta in metas])
        preds = (edge_probs >= args.threshold).float()
        train_route_acc = float((preds == labels).float().mean().item())

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "threshold": args.threshold,
        "vocab_size": args.vocab_size,
        "max_seq_len": args.max_seq_len,
        "routes": {
            "reasoning": "reasoning_only",
            "latent": "translator_latent_only_gated",
            "latent_edge": [ROLE_TRANSLATION, ROLE_REASONING],
        },
    }
    torch.save(ckpt, out_dir / "cvae_router.pt")
    (out_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "reasoning_csv": args.reasoning_csv,
                "latent_csv": args.latent_csv,
                "label_policy": args.label_policy,
                "threshold": args.threshold,
                "train_route_accuracy": train_route_acc,
                "history": history,
                "labels": metas,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[eval] train_route_accuracy={train_route_acc:.3f}", flush=True)
    print(f"[OK] wrote {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_csv", required=True)
    parser.add_argument("--latent_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label_policy", choices=["latent_if_fix", "latent_if_correct", "oracle_accuracy"], default="latent_if_fix")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--query_dim", type=int, default=64)
    parser.add_argument("--graph_hidden_dim", type=int, default=64)
    parser.add_argument("--encoder_hidden_dim", type=int, default=128)
    parser.add_argument("--decoder_hidden_dim", type=int, default=128)
    parser.add_argument("--query_embed_dim", type=int, default=64)
    parser.add_argument("--lstm_hidden_dim", type=int, default=64)
    parser.add_argument("--query_layers", type=int, default=1)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--beta_max", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--cycle_length", type=int, default=20)
    parser.add_argument("--free_bits_lambda", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_bilstm", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
