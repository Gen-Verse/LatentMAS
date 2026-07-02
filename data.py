from typing import Dict, Iterable, List, Optional

from datasets import load_dataset

from utils import extract_gold, normalize_answer


def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    # Prefer the namespaced id on HF (openai/gsm8k). Fail fast if not available.
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


def load_aime2025(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("yentinglin/aime_2025", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_aime2024(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_arc_easy(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        # Map choices
        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        # Map answers
        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_arc_challenge(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_winogrande(
    split: str = "validation",
    subset: str = "winogrande_debiased",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("allenai/winogrande", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        ask_str = 'Pickout proper choice that fits the _ in the following sentence:'
        sentence = item["sentence"].strip()
        option1 = str(item["option1"]).strip()
        option2 = str(item["option2"]).strip()
        question = f"{ask_str}\n{sentence}\n1: {option1}\n2: {option2}"
        answer = str(item["answer"])
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_mbppplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/mbppplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
Your answer will be tested on test cases like:
{item["test_list"][0]}
{item["test_list"][1]}
{item["test_list"][2]}
"""

        answer = str(item["test"])
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_humanevalplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/humanevalplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
"""
        raw_answer = str(item["test"])
        answer = raw_answer.replace('candidate', item['entry_point'])
        answer += f'\n\ncheck({item["entry_point"]})'
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


# qa data from https://github.com/lupantech/AgentFlow/tree/main
from typing import Iterable, Dict, Optional
from datasets import load_dataset

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


def load_medqa(split=None, subset=None, cache_dir=None):

    ds = load_dataset("json", data_files="./data/medqa.json", split='train')
    for item in ds:
        question = item["query"]
        raw_answer = str(item["answer"])

        choice_map = {"0":"A", "1":"B", "2":"C", "3":"D"}

        for idx, op in enumerate(item['options']):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break

        gold = normalize_answer(answer)

        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


# juletxara/mgsm only ships these 11 configs upstream -- no Lao/Khmer/Burmese/Amharic
# data exists (verified via datasets.get_dataset_config_names("juletxara/mgsm")).
MGSM_SUPPORTED_LANGUAGES = frozenset({"bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"})


def load_mgsm(
    split: str = "test",
    lang: str = "en",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:

    #mgsm has each language as a config (like bn (bengali), en (english))
    #we want to run using a variable lang that specifies the language you want this to occur in
    if lang is None:
        raise ValueError("lang must be provided")
    if lang not in MGSM_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"MGSM does not have data for language '{lang}'. juletxara/mgsm only covers "
            f"{sorted(MGSM_SUPPORTED_LANGUAGES)}. This is an upstream dataset limitation "
            "(no Lao/Khmer/Burmese/Amharic release exists), not a config error."
        )
    ds = load_dataset("juletxara/mgsm", lang, split=split, cache_dir=cache_dir)

    for item in ds:
        #check for an explicit language field (more so that we can copy this for other datasets)
        item_lang = None
        if "language" in item:
            item_lang = item.get("language")
        elif "lang" in item:
            item_lang = item.get("lang")

        if item_lang is not None and str(item_lang).lower() != str(lang).lower():
            continue
        
        question = (
            item.get("question")
            or item.get("question_text")
            or item.get("prompt")
            or ""
        )
        question = str(question).strip()

        raw_answer = item.get("answer")
        if raw_answer is None or raw_answer == "":
            if "answer_number" in item and item.get("answer_number") is not None:
                raw_answer = str(item.get("answer_number"))
            elif "equation_solution" in item and item.get("equation_solution"):
                raw_answer = item.get("equation_solution")
            else:
                raw_answer = ""

        solution = str(raw_answer).strip()

        #gold answer for comparison
        if "answer_number" in item and item.get("answer_number") is not None:
            gold = normalize_answer(str(item.get("answer_number")))
        else:
            gold = normalize_answer(solution)

        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


# McGill-NLP/mgsm-pro keys language by HF *split* name (not config -- the two configs,
# "ic" and "symbolic", are instantiation categories, not languages). Verified via
# datasets.get_dataset_config_names/get_dataset_split_names. Language coverage is
# notably NOT the same 11 languages as base MGSM -- it includes Amharic/Igbo/Twi/Yoruba
# but not Bengali/German/Russian/Telugu/Thai.
MGSM_PRO_LANG_TO_SPLIT = {
    "am": "amharic", "zh": "chinese", "en": "english", "fr": "french",
    "ig": "igbo", "ja": "japanese", "sw": "swahili", "tw": "twi", "yo": "yoruba",
}


def load_mgsm_pro(
    lang: str = "en", config: str = "symbolic", cache_dir: Optional[str] = None
) -> Iterable[Dict]:
    if lang is None:
        raise ValueError("lang must be provided")
    split = MGSM_PRO_LANG_TO_SPLIT.get(lang)
    if split is None:
        raise ValueError(
            f"MGSM-Pro does not have data for language '{lang}'. It only covers "
            f"{sorted(MGSM_PRO_LANG_TO_SPLIT)}. This is an upstream dataset limitation "
            "(languages are exposed as dataset splits, not a config parameter)."
        )
    if config not in ("ic", "symbolic"):
        raise ValueError(f"MGSM-Pro config must be 'ic' or 'symbolic', got '{config}'.")
    ds = load_dataset("McGill-NLP/mgsm-pro", config, split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("question", "")).strip()
        solution = str(item.get("answer", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_belebele(split: str = "test", lang: str = "eng_Latn", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("facebook/belebele", lang, split=split, cache_dir=cache_dir)
    for item in ds:
        question_text = str(item.get("question", "")).strip()
        flores_passage = str(item.get("flores_passage", "")).strip()
        mcqa = f"{flores_passage}\n{question_text}\n1: {item.get('mc1_targets')}\n2: {item.get('mc2_targets')}\n3: {item.get('mc3_targets')}\n4: {item.get('mc4_targets')}"
        solution = str(item.get("correct_answer_num", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": mcqa, "solution": solution, "gold": gold}


# SEA-HELM is a framework spanning many separate component datasets, each published
# directly under aisingapore/<ComponentName> -- there is no single "sea-helm-<subset>"
# repo (verified: that ID pattern 404s; the real components are e.g.
# aisingapore/NLU-Belebele-MCQA, aisingapore/NLR-Causal-Reasoning,
# aisingapore/NLU-Question-Answering, aisingapore/Cultural-Evaluation-Kalahi). Most of
# these components are gated -- request access on the HF dataset page before use.
#
# Verified schema for NLU-Belebele-MCQA (2026-07-02, via load_dataset probe): the
# dataset is *language-configured*, not a flat "test" split -- e.g.
# load_dataset('aisingapore/NLU-Belebele-MCQA', 'th') with splits {"eval", "examples"}
# (895 / 5 rows respectively for th), each row's MCQA content nested one level under
# a single-element `prompts` list: {choice1..4, question, text}, with `label` (the
# correct choice letter) at the top level. Schema of other components is unverified --
# don't assume this shape generalizes without checking.
SEA_HELM_LANGUAGES = frozenset({
    "en", "id", "jv", "km", "lo", "ms", "my", "su", "ta", "th", "tl", "vi", "zh", "zh_t",
})


def load_sea_helm(
    lang: str = "th",
    subset: str = "NLU-Belebele-MCQA",
    split: str = "eval",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    if lang not in SEA_HELM_LANGUAGES:
        raise ValueError(
            f"SEA-HELM component '{subset}' has no config for language '{lang}'. "
            f"Available: {sorted(SEA_HELM_LANGUAGES)}."
        )
    ds = load_dataset(f"aisingapore/{subset}", lang, split=split, cache_dir=cache_dir)
    for item in ds:
        prompt = (item.get("prompts") or [{}])[0]
        passage = str(prompt.get("text", "")).strip()
        question_text = str(prompt.get("question", "")).strip()
        # `label` is a letter (A-D) mapping 1:1 to choice1..choice4 in order -- keep the
        # displayed choice keys as letters too so they match `solution` directly.
        choices = {
            letter: prompt.get(f"choice{i}")
            for i, letter in enumerate(("A", "B", "C", "D"), start=1)
        }
        ordered_lines = [f"{k}: {v}" for k, v in choices.items() if v is not None]
        question = "\n".join([passage, question_text, *ordered_lines]).strip()
        solution = str(item.get("label", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


# Verified schema (2026-07-02, via load_dataset probe): mahbubhimel/MathMist has a
# single "default" config; there is no "test" split -- each *language* is its own
# split (e.g. "amharic", "swahili", "nctb_corpus_bangla" ...), and columns are
# capitalized ("Question", "Exact Answer", not "question"/"answer"). Of this project's
# target languages, only Amharic and Swahili are covered.
MATHMIST_LANG_TO_SPLIT = {"am": "amharic", "sw": "swahili"}


def load_mathmist(lang: str = "am", subset: str = "default", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    split = MATHMIST_LANG_TO_SPLIT.get(lang)
    if split is None:
        raise ValueError(
            f"MathMist does not have data for language '{lang}'. It only covers "
            f"{sorted(MATHMIST_LANG_TO_SPLIT)} of this project's target set "
            "(the paper's 13-language claim does not hold up on the actual dataset)."
        )
    ds = load_dataset("mahbubhimel/MathMist", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("Question", "")).strip()
        solution = str(item.get("Exact Answer", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_banglamath(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("kawchar85/Bangla-Math", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("question", "")).strip()
        solution = str(item.get("answer", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


# BAAI/LaoBench (Dec 2025, apache-2.0, single "test" split, 7000 rows) closes the
# reasoning-benchmark gap for Lao -- MGSM/MRG have none. Verified schema: a `data_type`
# column distinguishes 4-way MCQA rows ("K12 Foundational Education",
# "Knowledge Application" -- answer is a letter A-D, options in columns A/B/C/D) from
# "Bilingual Translation" rows (question in Chinese, answer in Lao, A-D all None). Only
# the MCQA subset is reasoning-evaluable; translation rows are excluded by default.
LAOBENCH_MCQA_TYPES = frozenset({"K12 Foundational Education", "Knowledge Application"})


def load_laobench(
    split: str = "test",
    data_types: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("BAAI/LaoBench", split=split, cache_dir=cache_dir)
    wanted = set(data_types) if data_types else set(LAOBENCH_MCQA_TYPES)
    for item in ds:
        if item.get("data_type") not in wanted:
            continue
        stem = str(item.get("question", "")).strip()
        options = {k: item.get(k) for k in ("A", "B", "C", "D") if item.get(k) is not None}
        ordered_lines = [f"{k}: {v}" for k, v in options.items()]
        question = stem + "\n" + "\n".join(ordered_lines) if ordered_lines else stem
        answer = str(item.get("answer", "")).strip()
        gold = normalize_answer(answer)
        yield {"question": question, "solution": answer, "gold": gold}


# MauroPello/multilingual-reasoning-gym-sft only ships these 13 configs upstream --
# no Thai/Lao/Khmer/Burmese/Amharic/Swahili data exists (verified via
# datasets.get_dataset_config_names(...)).
MRG_SUPPORTED_LANGUAGES = frozenset({"de", "en", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ru", "uk", "zh"})


def load_multilingual_reasoning_gym(split: str = "train", lang: str = "en", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    if lang not in MRG_SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Multilingual Reasoning Gym does not have data for language '{lang}'. "
            f"MauroPello/multilingual-reasoning-gym-sft only covers {sorted(MRG_SUPPORTED_LANGUAGES)}. "
            "This is an upstream dataset limitation (no SEA-language release exists), not a "
            "config error -- use Belebele, FLORES+, or SEA-Vision for SEA target languages instead."
        )
    ds = load_dataset("MauroPello/multilingual-reasoning-gym-sft", lang, split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("instruction", item.get("question", ""))).strip()
        solution = str(item.get("output", item.get("answer", ""))).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_seabench(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("SeaLLMs/SeaBench-Audio", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("question", "")).strip()
        solution = str(item.get("answer", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_multichallenge(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("ScaleAI/MultiChallenge", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("prompt", "")).strip()
        solution = str(item.get("response", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_xquad(split: str = "validation", lang: str = "en", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("xquad", f"xquad.{lang}", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("context", "")) + "\n" + str(item.get("question", "")).strip()
        answers = item.get("answers", {}).get("text", [])
        solution = str(answers[0] if answers else "").strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_mlqa(split: str = "test", lang: str = "en", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("mlqa", f"mlqa.{lang}.{lang}", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("context", "")) + "\n" + str(item.get("question", "")).strip()
        answers = item.get("answers", {}).get("text", [])
        solution = str(answers[0] if answers else "").strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_flores_plus(split: str = "dev", lang: str = "eng_Latn", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("facebook/flores", lang, split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("sentence", "")).strip()
        solution = str(item.get("sentence", "")).strip()  # Depends on task (usually translation requires parallel target)
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}
