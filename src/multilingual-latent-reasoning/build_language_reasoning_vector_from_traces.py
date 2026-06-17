import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def load_payload(path: Path) -> Dict:
    with path.open("rb") as f:
        return pickle.load(f)


def mean_hidden_by_language(payload: Dict, roles: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for lang in payload["languages"]:
        chunks = []
        for ex in payload["traces"][lang]:
            for role in roles:
                agent = ex["agents"].get(role)
                if agent is None:
                    continue
                hidden = np.asarray(agent["hidden"], dtype=np.float32)
                if hidden.ndim != 3:
                    raise ValueError(f"Expected hidden shape [steps, layers, dim], got {hidden.shape}")
                chunks.append(hidden)
        if not chunks:
            raise ValueError(f"No hidden states found for language {lang} and roles {roles}")
        stacked = np.concatenate(chunks, axis=0)
        out[lang] = stacked.mean(axis=0)
    return out


def build_aligner(rank: int, lang_emb: Dict[str, np.ndarray]) -> np.ndarray:
    lang_mean_emb = {lang: np.mean(emb, axis=0) for lang, emb in lang_emb.items()}
    w = np.stack(list(lang_mean_emb.values())).T
    _, language_count = w.shape

    wc = w @ np.ones(language_count) / language_count
    u, s, vh = np.linalg.svd(w - wc.reshape(-1, 1) @ np.ones((1, language_count)), full_matrices=False)
    ws = u[:, :rank]
    gamma = vh.T[:, :rank] @ np.diag(s[:rank])
    best_fit_w = wc.reshape(-1, 1) @ np.ones((1, language_count)) + ws @ gamma.T

    wc_new = np.linalg.pinv(best_fit_w).T @ np.ones(language_count)
    wc_new /= (wc_new ** 2).sum()
    prod = best_fit_w - wc_new.reshape(-1, 1) @ np.ones((1, language_count))

    u, _, _ = np.linalg.svd(prod, full_matrices=False)
    return u[:, :rank].T


def build_vector(lang_layer_means: Dict[str, np.ndarray], rank: int) -> torch.Tensor:
    langs = list(lang_layer_means)
    layer_count = lang_layer_means[langs[0]].shape[0]
    vectors = []
    for layer_idx in range(layer_count):
        layer_emb = {
            lang: lang_layer_means[lang][layer_idx]
            for lang in langs
        }
        vectors.append(torch.tensor(build_aligner(rank, layer_emb), dtype=torch.float32))
    return torch.stack(vectors, dim=0)


def parse_roles(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to latent_mas_mgsm_batch_traces.pkl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write vector.pt.",
    )
    parser.add_argument(
        "--roles",
        default="planner,critic,refiner",
        help="Comma-separated agent roles used to estimate the language subspace.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=-1,
        help="Language subspace rank. Default is number of languages - 1.",
    )
    args = parser.parse_args()

    payload = load_payload(Path(args.input))
    roles = parse_roles(args.roles)
    rank = args.rank if args.rank > 0 else len(payload["languages"]) - 1
    lang_means = mean_hidden_by_language(payload, roles)
    vector = build_vector(lang_means, rank)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vector, out_path)

    summary = {
        "input": args.input,
        "output": args.output,
        "languages": payload["languages"],
        "roles": roles,
        "rank": rank,
        "vector_shape": list(vector.shape),
        "definition": (
            "Per-layer language subspace estimated from mean LatentMAS hidden states "
            "using the SVD construction from Language-Reasoning Disentangle."
        ),
    }
    with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
