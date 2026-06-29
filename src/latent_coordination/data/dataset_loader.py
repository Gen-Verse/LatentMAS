"""Dataset loader for SEA-Vision and SEA-VL datasets using Meta's Belebele benchmark."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)

# SEA-VL records language as a human name + non-ISO-639-1 code, e.g. "Myanmar (mm)",
# "Thai (tha)", "Indonesian (ind)". Map the leading name → ISO-639-1 so it composes with
# the rest of the pipeline (which keys on th/my/km/lo/...). SEA-VL is SEA-only, so non-SEA
# targets like Amharic/Swahili are simply absent (expected, not an error).
_SEAVL_NAME_TO_ISO = {
    "thai": "th", "burmese": "my", "myanmar": "my", "khmer": "km", "lao": "lo",
    "vietnamese": "vi", "indonesian": "id", "malay": "ms", "filipino": "fil",
    "tagalog": "fil", "tamil": "ta", "chinese": "zh", "english": "en", "javanese": "jv",
    "sundanese": "su", "cebuano": "ceb",
}


@dataclass
class Sample:
    """A standard representation engineering dataset sample."""
    sample_id: str
    language: str
    text: str
    question: str
    reference_answer: str
    image_path: Optional[str] = None


class DatasetLoader:
    """Loads samples from the massively multilingual Belebele dataset for representation analyses."""

    def __init__(self, cache_dir: Optional[str] = ".cache/datasets") -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DatasetLoader initialized with cache_dir=%s", cache_dir)

    def load_belebele(
        self,
        languages: List[str],
        split: str = "test",
        max_per_language: Optional[int] = None,
    ) -> List[Sample]:
        """Load Meta's Belebele benchmark dataset."""
        from datasets import load_dataset

        BELEBELE_LANG_MAP = {
            "th": "tha_Thai", "my": "mya_Mymr", "km": "khm_Khmr", "lo": "lao_Laoo",
            "jv": "jav_Latn", "su": "sun_Latn", "ceb": "ceb_Latn", "vi": "vie_Latn",
            "id": "ind_Latn", "ms": "zsm_Latn", "fil": "tgl_Latn", "am": "amh_Ethi",
            "sw": "swh_Latn",
        }

        samples: List[Sample] = []
        for lang in languages:
            config_name = BELEBELE_LANG_MAP.get(lang, "eng_Latn")
            logger.info("Fetching real Belebele dataset config %s for language %s", config_name, lang)
            try:
                ds = load_dataset("facebook/belebele", config_name, split="test")
                count = 0
                for item in ds:
                    if max_per_language and count >= max_per_language:
                        break

                    passage = item.get("flores_passage", "")
                    question = item.get("question", "")
                    opt1 = item.get("mc_answer1", "")
                    opt2 = item.get("mc_answer2", "")
                    opt3 = item.get("mc_answer3", "")
                    opt4 = item.get("mc_answer4", "")
                    correct = item.get("correct_answer_num", "1")

                    formatted_text = (
                        f"Passage: {passage}\n\nQuestion: {question}\nOptions:\n"
                        f"1. {opt1}\n2. {opt2}\n3. {opt3}\n4. {opt4}\n\n"
                        f"Identify the correct option number (1, 2, 3, or 4). Answer:"
                    )

                    samples.append(Sample(
                        sample_id=f"belebele_{lang}_{item.get('question_number', count)}",
                        language=lang, text=formatted_text, question=question,
                        reference_answer=correct, image_path=None
                    ))
                    count += 1
            except Exception as e:
                logger.error("Failed to load Belebele dataset for %s: %s", lang, e)
                raise RuntimeError(f"Strict evaluation mode: failed to load real Belebele dataset for language {lang}. Error: {e}") from e

        return samples

    def load_sea_vision(
        self,
        languages: List[str],
        split: str = "test",
        max_per_language: Optional[int] = None,
        local_dir: Optional[str] = None,
    ) -> List[Sample]:
        """Load SEA-Vision benchmark dataset (document parsing & text-centric reasoning)."""
        from huggingface_hub import hf_hub_download
        import json

        logger.info("Fetching real SEA-Vision dataset from HuggingFace.")
        try:
            path = hf_hub_download(repo_id='xingranzhao/SEA-Vision', filename='all_qa_data.jsonl', repo_type='dataset')
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download xingranzhao/SEA-Vision/all_qa_data.jsonl from HuggingFace. Error: {exc}. "
                "Mock sample generation is not permitted."
            ) from exc

        lang_map = {
            "泰语": "th", "缅甸语": "my", "高棉语": "km", "老挝语": "lo",
            "越南语": "vi", "印尼语": "id", "马来语": "ms", "菲律宾语": "fil",
            "英语": "en", "中文": "zh",
        }

        samples: List[Sample] = []
        with open(path, "r", encoding="utf-8") as f:
            by_lang = {}
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                zh_lang = row.get("语言", "")
                iso_lang = lang_map.get(zh_lang)
                if iso_lang:
                    by_lang.setdefault(iso_lang, []).append(row)

            for lang in languages:
                rows = by_lang.get(lang, [])
                for i, row in enumerate(rows):
                    if max_per_language and i >= max_per_language:
                        break
                    samples.append(Sample(
                        sample_id=f"seavision_{lang}_{row.get('数据索引', i)}",
                        language=lang,
                        text=row.get("最终问题", ""),
                        question=row.get("最终问题", ""),
                        reference_answer=row.get("最终答案", ""),
                        image_path=row.get("图片编号")
                    ))

        return samples

    def load_sea_vl(
        self,
        languages: List[str],
        split: str = "train",          # SEACrowd/sea-vl_crowdsourcing only publishes 'train'
        max_per_language: Optional[int] = None,
    ) -> List[Sample]:
        """Load SEA-VL native-language captions (Multicultural Grounding).

        Schema: ``caption_native_lang`` is the native-language text, ``native_lang`` is a
        "Name (code)" string (e.g. "Myanmar (mm)"). We map the name → ISO-639-1, use the
        native caption as the sample text, and cap per language. No mock data; languages with
        no rows simply yield nothing (SEA-VL is SEA-only, so e.g. am/sw are legitimately absent).
        """
        logger.info("Fetching real SEA-VL dataset.")
        from datasets import load_dataset

        wanted = set(languages) if languages else None
        per_lang_count: Dict[str, int] = {}
        try:
            ds = load_dataset("SEACrowd/sea-vl_crowdsourcing", split=split)
            samples = []
            for i, item in enumerate(ds):
                raw = (item.get("native_lang") or "").strip()
                name = raw.split("(")[0].strip().lower()
                lang = _SEAVL_NAME_TO_ISO.get(name)
                if lang is None or (wanted and lang not in wanted):
                    continue
                if max_per_language and per_lang_count.get(lang, 0) >= max_per_language:
                    continue
                text = (item.get("caption_native_lang") or item.get("caption") or "").strip()
                if not text:
                    continue
                per_lang_count[lang] = per_lang_count.get(lang, 0) + 1
                samples.append(Sample(
                    sample_id=f"seavl_{lang}_{i}",
                    language=lang,
                    text=text,
                    question=text,
                    reference_answer=item.get("caption", ""),   # English caption as reference
                    image_path=None,
                ))
            if not samples:
                logger.warning(
                    "SEA-VL yielded 0 samples for languages %s (none of the requested "
                    "languages appear in this SEA dataset).", sorted(wanted) if wanted else "all",
                )
            return samples
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load real SEA-VL dataset from HuggingFace. Error: {exc}. "
                "No mock data fallback is permitted."
            ) from exc

    def load_sea_safeguardbench(
        self,
        languages: Optional[List[str]] = None,
        split: str = "test",
        max_samples: Optional[int] = None,
        repo_id: Optional[str] = None,
    ) -> List[Sample]:
        """Load a SEA safety-alignment benchmark.

        The original ``SeaEval/SEA-SafeguardBench`` ID is no longer published on the HF
        Hub, so the dataset repo is configurable. Pass ``repo_id`` (or set
        ``benchmarks.sea_safeguardbench.repo_id`` in the YAML) to a real, accessible
        safety dataset. No ID is hard-coded to a non-existent repo and no mock data is
        ever fabricated.

        Args:
            languages: ISO codes to filter; None loads all languages.
            split: Dataset split (typically "test").
            max_samples: Cap on total samples loaded.
            repo_id: HF dataset repo id for the safety benchmark (required).

        Returns:
            List of Sample objects with safety prompts and expected verdicts.
        """
        if not repo_id:
            raise ValueError(
                "load_sea_safeguardbench requires an explicit `repo_id` (or "
                "`benchmarks.sea_safeguardbench.repo_id` in config). The legacy "
                "'SeaEval/SEA-SafeguardBench' dataset is no longer available on the HF Hub; "
                "supply a verified safety dataset id to enable this benchmark."
            )
        logger.info("Fetching real SEA safety dataset '%s'.", repo_id)
        from datasets import load_dataset

        try:
            ds = load_dataset(repo_id, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load safety dataset '{repo_id}' from HuggingFace. Error: {exc}. "
                "No mock data fallback is permitted."
            ) from exc

        samples: List[Sample] = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            lang = item.get("language", "en")
            if languages and lang not in languages:
                continue
            samples.append(Sample(
                sample_id=f"safeguard_{lang}_{i}",
                language=lang,
                text=item.get("prompt", item.get("text", "")),
                question=item.get("prompt", item.get("text", "")),
                reference_answer=item.get("label", item.get("verdict", "safe")),
                image_path=None,
            ))

        logger.info("Loaded %d samples from SEA-SafeguardBench.", len(samples))
        return samples

    def load_mgsm(
        self,
        languages: List[str],
        split: str = "test",
        max_per_language: Optional[int] = None,
    ) -> List[Sample]:
        """Load the Multilingual Grade School Math (MGSM) benchmark."""
        from datasets import load_dataset

        MGSM_LANG_MAP = {
            "th": "th", "te": "te", "sw": "sw", "zh": "zh", "en": "en",
            "bn": "bn", "ja": "ja", "ru": "ru", "fr": "fr", "de": "de", "es": "es"
        }

        samples: List[Sample] = []
        for lang in languages:
            config_name = MGSM_LANG_MAP.get(lang)
            if not config_name:
                logger.warning("MGSM does not officially support %s; skipping.", lang)
                continue
            logger.info("Fetching real MGSM dataset config %s for language %s", config_name, lang)
            try:
                ds = load_dataset("juletxara/mgsm", config_name, split=split)
                count = 0
                for item in ds:
                    if max_per_language and count >= max_per_language:
                        break
                    
                    question = item.get("question", "")
                    correct = str(item.get("answer_number", item.get("answer", "")))
                    formatted_text = f"Question: {question}\nAnswer:"
                    
                    samples.append(Sample(
                        sample_id=f"mgsm_{lang}_{count}",
                        language=lang, text=formatted_text, question=question,
                        reference_answer=correct, image_path=None
                    ))
                    count += 1
            except Exception as e:
                logger.error("Failed to load MGSM dataset for %s: %s", lang, e)
                raise RuntimeError(f"Strict eval: failed to load real MGSM for {lang}.") from e

        return samples

    def load_global_mmlu(
        self,
        languages: List[str],
        split: str = "test",
        max_per_language: Optional[int] = None,
    ) -> List[Sample]:
        """Load the Global-MMLU benchmark."""
        from datasets import load_dataset

        GLOBAL_MMLU_MAP = {
            "ar": "ar", "bn": "bn", "de": "de", "es": "es", "fr": "fr",
            "hi": "hi", "id": "id", "it": "it", "ja": "ja", "ko": "ko",
            "pt": "pt", "ru": "ru", "sw": "sw", "te": "te", "th": "th",
            "yo": "yo", "zh": "zh"
        }

        samples: List[Sample] = []
        for lang in languages:
            config_name = GLOBAL_MMLU_MAP.get(lang)
            if not config_name:
                logger.warning("Global-MMLU does not officially support %s; skipping.", lang)
                continue
            logger.info("Fetching real Global-MMLU dataset config %s for language %s", config_name, lang)
            try:
                ds = load_dataset("CohereForAI/Global-MMLU", config_name, split=split)
                count = 0
                for item in ds:
                    if max_per_language and count >= max_per_language:
                        break
                    
                    question = item.get("question", "")
                    opt_A = item.get("option_a", "")
                    opt_B = item.get("option_b", "")
                    opt_C = item.get("option_c", "")
                    opt_D = item.get("option_d", "")
                    correct = item.get("answer", "")
                    
                    # Convert letter A-D to 1-4 for our common log-likelihood scoring
                    ans_map = {"A": "1", "B": "2", "C": "3", "D": "4"}
                    correct_num = ans_map.get(correct, correct)
                    
                    formatted_text = (
                        f"Question: {question}\nOptions:\n"
                        f"1. {opt_A}\n2. {opt_B}\n3. {opt_C}\n4. {opt_D}\n\n"
                        f"Identify the correct option number (1, 2, 3, or 4). Answer:"
                    )
                    
                    samples.append(Sample(
                        sample_id=f"gmmlu_{lang}_{count}",
                        language=lang, text=formatted_text, question=question,
                        reference_answer=correct_num, image_path=None
                    ))
                    count += 1
            except Exception as e:
                logger.error("Failed to load Global-MMLU dataset for %s: %s", lang, e)
                raise RuntimeError(f"Strict eval: failed to load real Global-MMLU for {lang}.") from e

        return samples
