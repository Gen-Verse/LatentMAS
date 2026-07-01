from typing import Dict, Iterable, Optional

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


def load_mgsm(
    split: str = "test",
    lang: str = "en",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:

    #mgsm has each language as a config (like bn (bengali), en (english))
    #we want to run using a variable lang that specifies the language you want this to occur in
    if lang is None:
        raise ValueError("lang must be provided")
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


def load_mgsm_pro(split: str = "test", lang: str = "en", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    if lang is None:
        raise ValueError("lang must be provided")
    ds = load_dataset("McGill-NLP/mgsm-pro", lang, split=split, cache_dir=cache_dir)
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


def load_sea_helm(subset: str = "NLU-Belebele-MCQA", split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset(f"aisingapore/sea-helm-{subset}", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("question", item.get("prompt", ""))).strip()
        solution = str(item.get("answer", item.get("label", ""))).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_mathmist(split: str = "test", subset: str = "default", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("mahbubhimel/MathMist", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("question", "")).strip()
        solution = str(item.get("answer", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_banglamath(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("kawchar85/Bangla-Math", split=split, cache_dir=cache_dir)
    for item in ds:
        question = str(item.get("question", "")).strip()
        solution = str(item.get("answer", "")).strip()
        gold = normalize_answer(solution)
        yield {"question": question, "solution": solution, "gold": gold}


def load_multilingual_reasoning_gym(split: str = "train", lang: str = "en", cache_dir: Optional[str] = None) -> Iterable[Dict]:
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
