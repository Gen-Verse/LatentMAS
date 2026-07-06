#!/usr/bin/env python
"""Export precomputed Geo_L per-language risk profiles for Module D.

Lives on the MECHANISTIC side of the firewall (strategy.md §6): all activation
extraction and geometric analysis happens here via ``mechanistic_disentangle``
utilities, and the result crosses to ``latent_coordination`` as a plain JSON
artifact consumed by ``latent_coordination.topology.geo_profile.GeoProfile``.

Features per language (compressed summary, strategy.md §4.2 — 3–8 scalars,
NOT a raw per-layer concatenation):

  1. late_layer_cka_to_english : mean linear-CKA between the target language's
     and English's activations over the late-layer band (0.65L–0.9L).
  2. clap_dealignment          : CLAP delta between English and target
     activations at the mid-stack layer (~0.65L).
  3. norm_distortion_ratio     : mean ||h_en|| / ||h_tgt|| at the same layer
     (the Magnitude Distortion Paradox diagnostic).

Requires a GPU-capable environment for real 7-8B backbones (a small model works
on CPU for smoke runs). Real FLORES+ parallel text only — no synthetic prompts.

Usage:
    python scripts/export_geo_profiles.py \
        --model aisingapore/Llama-SEA-LION-v3-8B-IT \
        --languages th,lo,km,my,am,sw \
        --n-samples 64 \
        --output results/mechanistic/geo_profiles.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

logger = logging.getLogger(__name__)

FLORES_CODES = {
    "th": "tha_Thai", "my": "mya_Mymr", "km": "khm_Khmr", "lo": "lao_Laoo",
    "am": "amh_Ethi", "sw": "swh_Latn", "bn": "ben_Beng", "te": "tel_Telu",
    "en": "eng_Latn",
}

FEATURE_NAMES = [
    "late_layer_cka_to_english",
    "clap_dealignment",
    "norm_distortion_ratio",
]


def _load_flores(lang_code: str, n: int):
    from datasets import load_dataset
    ds = load_dataset("openlanguagedata/flores_plus", name=lang_code, split="devtest")
    return [ds[i]["text"] for i in range(min(n, len(ds)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id for activation extraction.")
    parser.add_argument("--languages", default="th,lo,km,my,am,sw",
                        help="Comma-separated ISO-639-1 codes (must be in FLORES_CODES).")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--output", default="results/mechanistic/geo_profiles.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    unknown = [l for l in langs if l not in FLORES_CODES or l == "en"]
    if unknown:
        raise SystemExit(f"Unsupported language(s) {unknown}; known: {sorted(FLORES_CODES)} (en is the anchor).")

    from mechanistic_disentangle.geometry.activation_extractor import ActivationExtractor
    from shared.metrics import compute_cka_alignment, compute_clap
    from shared.model_loader import ModelLoadSpec, load_model_and_tokenizer

    spec = ModelLoadSpec(
        model_id=args.model, device=args.device,
        load_in_8bit=args.load_in_8bit, output_hidden_states=True,
    )
    model, tokenizer = load_model_and_tokenizer(spec)
    extractor = ActivationExtractor(model=model, tokenizer=tokenizer, device=args.device)

    n_layers = extractor.n_layers
    mid_layer = int(0.65 * n_layers)
    late_band = list(range(mid_layer, max(mid_layer + 1, int(0.9 * n_layers))))
    layer_ids = sorted(set([mid_layer] + late_band))
    logger.info("Model has %d layers; mid=%d, late band=%s", n_layers, mid_layer, late_band)

    en_texts = _load_flores(FLORES_CODES["en"], args.n_samples)
    en_acts = extractor.extract(en_texts, layer_ids=layer_ids)

    profiles = {}
    for lang in langs:
        texts = _load_flores(FLORES_CODES[lang], args.n_samples)
        acts = extractor.extract(texts, layer_ids=layer_ids)

        ckas = [
            compute_cka_alignment(acts[lid].float(), en_acts[lid].float())
            for lid in late_band if lid in acts and lid in en_acts
        ]
        late_cka = float(sum(ckas) / len(ckas)) if ckas else float("nan")
        clap = float(compute_clap(en_acts[mid_layer].float(), acts[mid_layer].float()))
        norm_ratio = float(
            (en_acts[mid_layer].norm(dim=-1).mean() / acts[mid_layer].norm(dim=-1).mean().clamp(min=1e-9)).item()
        )
        profiles[lang] = [late_cka, clap, norm_ratio]
        logger.info("Geo_L[%s] = %s", lang, profiles[lang])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "feature_names": FEATURE_NAMES,
        "model": args.model,
        "n_samples": args.n_samples,
        "profiles": profiles,
    }
    with out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    logger.info("Geo_L artifact written to %s (%d languages x %d features).",
                out, len(profiles), len(FEATURE_NAMES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
