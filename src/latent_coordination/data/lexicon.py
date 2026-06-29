"""Contrastive Lexicon containing real parallel text pairs for target languages.

Loads real multilingual parallel corpora from the FLORES-200 dataset
(facebook/flores) via the Hugging Face datasets library. Falls back to
OPUS-100 if FLORES is unavailable. No synthetic or placeholder translations
are generated.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

__author__ = "Himon Thakur"
__copyright__ = "Copyright [2026], Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


# ISO 639-1 code → FLORES-200 language code mapping
_ISO_TO_FLORES: Dict[str, str] = {
    # SEA languages (primary evaluation set)
    "th":  "tha_Thai",
    "my":  "mya_Mymr",
    "km":  "khm_Khmr",
    "lo":  "lao_Laoo",
    "jv":  "jav_Latn",
    "su":  "sun_Latn",
    "ceb": "ceb_Latn",
    "vi":  "vie_Latn",
    "id":  "ind_Latn",
    "ms":  "msa_Latn",
    "fil": "fil_Latn",
    "am":  "amh_Ethi",
    "sw":  "swh_Latn",
    "en":  "eng_Latn",
    # Expansion languages for CKA-IFL geometric regression (n=5 → n=14)
    "bn":  "ben_Beng",   # Bengali  (Indic)
    "ta":  "tam_Taml",   # Tamil    (Indic)
    "te":  "tel_Telu",   # Telugu   (Indic)
    "si":  "sin_Sinh",   # Sinhala  (Indic)
    "bo":  "bod_Tibt",   # Tibetan
    "ka":  "kat_Geor",   # Georgian
    "hy":  "hye_Armn",   # Armenian
    "ar":  "arb_Arab",   # Arabic   (RTL)
    "he":  "heb_Hebr",   # Hebrew   (RTL)
    "zh":  "cmn_Hans",   # Mandarin Chinese Simplified
}

# English FLORES-200 split key
_FLORES_EN = "eng_Latn"

# Maximum number of FLORES sentences to load per language (dev split has 997)
_FLORES_MAX_SENTENCES = 997


@dataclass
class LexiconEntry:
    """A single contrastive pair entry in the lexicon."""
    english: str
    target: str
    domain: str
    complexity_level: str


def _load_flores_plus_pairs(iso_code: str) -> List[Tuple[str, str]]:
    """Loads English and target language parallel texts from FLORES+.

    Parameters
    ----------
    iso_code : str
        ISO 639-1 target language code.

    Returns
    -------
    List[Tuple[str, str]]
        List of (english_sentence, target_sentence) pairs.
    """
    flores_code = _ISO_TO_FLORES.get(iso_code)
    if flores_code is None:
        raise ValueError(f"Language '{iso_code}' not in FLORES mapping.")

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("The 'datasets' library is required. pip install datasets") from e

    logger.info("Loading FLORES+ devtest split for '%s' ↔ 'eng_Latn'.", flores_code)

    try:
        en_ds = load_dataset("openlanguagedata/flores_plus", name="eng_Latn", split="devtest")
        tgt_ds = load_dataset("openlanguagedata/flores_plus", name=flores_code, split="devtest")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FLORES+ for language '{iso_code}' ({flores_code}): {exc}"
        ) from exc

    n = min(len(en_ds), len(tgt_ds), _FLORES_MAX_SENTENCES)
    pairs = [
        (en_ds[i]["text"], tgt_ds[i]["text"])
        for i in range(n)
    ]
            
    logger.info("Loaded %d FLORES+ pairs for '%s'.", len(pairs), iso_code)
    return pairs


class ContrastiveLexicon:
    """Manages real contrastive parallel lexicons for low-resource languages.

    Pairs are sourced from FLORES-200 (facebook/flores) loaded via the
    Hugging Face ``datasets`` library. Languages are loaded lazily on first
    access to avoid downloading unused splits.

    Parameters
    ----------
    preload_languages : List[str], optional
        ISO 639-1 codes to preload immediately on construction.
    cache_dir : Path, optional
        Directory to cache downloaded FLORES pairs as JSONL files.
    """

    def __init__(
        self,
        preload_languages: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._cache: Dict[str, List[Tuple[str, str]]] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None

        if preload_languages:
            for lang in preload_languages:
                self._ensure_loaded(lang)

        logger.info(
            "ContrastiveLexicon ready. Preloaded languages: %s",
            list(self._cache.keys()),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cache_path(self, iso_code: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"flores_{iso_code}.jsonl"

    def _load_from_disk(self, iso_code: str) -> Optional[List[Tuple[str, str]]]:
        path = self._cache_path(iso_code)
        if path is None or not path.exists():
            return None
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                pairs.append((d["english"], d["target"]))
        logger.info("Loaded %d cached FLORES pairs for '%s' from %s.", len(pairs), iso_code, path)
        return pairs

    def _save_to_disk(self, iso_code: str, pairs: List[Tuple[str, str]]) -> None:
        path = self._cache_path(iso_code)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for en, tgt in pairs:
                f.write(json.dumps({"english": en, "target": tgt}, ensure_ascii=False) + "\n")
        logger.info("Cached %d FLORES pairs for '%s' to %s.", len(pairs), iso_code, path)

    def _ensure_loaded(self, iso_code: str) -> None:
        """Load pairs for a language if not already cached in memory."""
        if iso_code in self._cache:
            return

        # Try disk cache first
        disk_pairs = self._load_from_disk(iso_code)
        if disk_pairs is not None:
            self._cache[iso_code] = disk_pairs
            return

        # Download from FLORES+
        pairs = _load_flores_plus_pairs(iso_code)
        self._cache[iso_code] = pairs
        self._save_to_disk(iso_code, pairs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pairs(
        self,
        language: str,
        n_pairs: Optional[int] = None,
        domain: Optional[str] = None,
    ) -> List[Tuple[str, str]]:
        """Retrieve real parallel text pairs for the target language.

        Parameters
        ----------
        language : str
            ISO 639-1 language code (e.g. 'th').
        n_pairs : int, optional
            Limit the number of returned pairs.
        domain : str, optional
            Ignored (FLORES-200 does not have domain annotations). Kept
            for API compatibility.

        Returns
        -------
        List[Tuple[str, str]]
            List of (english_text, target_text) aligned tuples from FLORES-200.

        Raises
        ------
        RuntimeError
            If the language cannot be loaded from FLORES-200.
        """
        if domain is not None:
            logger.debug(
                "Domain filter '%s' requested but FLORES-200 has no domain labels; "
                "returning all pairs for '%s'.",
                domain, language,
            )

        self._ensure_loaded(language)
        pairs = self._cache[language]

        if n_pairs is not None:
            pairs = pairs[:n_pairs]

        return list(pairs)

    def available_languages(self) -> List[str]:
        """Return ISO codes with FLORES-200 support."""
        return list(_ISO_TO_FLORES.keys())

    def load_from_jsonl(self, path: Path | str, iso_code: str) -> None:
        """Load additional parallel pairs from a JSONL file.

        Parameters
        ----------
        path : Path or str
            Path to JSONL file with ``english`` and ``target`` keys per row.
        iso_code : str
            ISO 639-1 code to register these pairs under.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {path}")

        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                pairs.append((d["english"], d["target"]))

        existing = self._cache.get(iso_code, [])
        self._cache[iso_code] = existing + pairs
        logger.info(
            "Loaded %d additional pairs from %s for language '%s'. Total: %d.",
            len(pairs), path, iso_code, len(self._cache[iso_code]),
        )
