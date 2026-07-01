
import os
import json

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


# Load optional prompt translations (per-language) from assets/prompt_translations.json
_TRANSLATIONS = {}
try:
    _p = os.path.join(os.path.dirname(__file__), "assets", "prompt_translations.json")
    if os.path.exists(_p):
        with open(_p, "r", encoding="utf-8") as _f:
            _TRANSLATIONS = json.load(_f)
except Exception:
    _TRANSLATIONS = {}


_LANGUAGE_DIRECTIVES = {
    "en": "Respond entirely in English, including any reasoning inside <think> tags.",
    "bn": "<think> ট্যাগের ভিতরের যুক্তিসহ সম্পূর্ণভাবে বাংলায় উত্তর দিন।",
    "de": "Antworten Sie vollständig auf Deutsch, einschließlich aller Überlegungen innerhalb von <think>-Tags.",
    "es": "Responde completamente en español, incluido cualquier razonamiento dentro de las etiquetas <think>.",
    "fr": "Répondez entièrement en français, y compris tout raisonnement à l'intérieur des balises <think>.",
    "ja": "<think>タグ内の推論も含めて、回答はすべて日本語で書いてください。",
    "ru": "Отвечайте полностью на русском языке, включая любые рассуждения внутри тегов <think>.",
    "sw": "Jibu kabisa kwa Kiswahili, ikiwemo hoja yoyote ndani ya tagi za <think>.",
    "te": "<think> ట్యాగ్‌లలోని ఏ తర్కాన్నైనా కలుపుకుని పూర్తిగా తెలుగులో సమాధానం ఇవ్వండి.",
    "th": "ตอบเป็นภาษาไทยทั้งหมด รวมถึงการให้เหตุผลภายในแท็ก <think>",
    "zh": "请完全用中文回答，包括 <think> 标签内的任何推理。",
}


_THINK_PREFILLS = {
    "en": "<think>\nLet's reason step by step in English.\n",
    "bn": "<think>\nচলুন বাংলায় ধাপে ধাপে যুক্তি করি।\n",
    "de": "<think>\nLassen Sie uns Schritt für Schritt auf Deutsch überlegen.\n",
    "es": "<think>\nRazonemos paso a paso en español.\n",
    "fr": "<think>\nRaisonnons étape par étape en français.\n",
    "ja": "<think>\n日本語で段階的に考えましょう。\n",
    "ru": "<think>\nДавайте рассуждать шаг за шагом на русском языке.\n",
    "sw": "<think>\nTufikiri hatua kwa hatua kwa Kiswahili.\n",
    "te": "<think>\nతెలుగులో దశలవారీగా ఆలోచిద్దాం.\n",
    "th": "<think>\nมาคิดทีละขั้นตอนเป็นภาษาไทยกัน\n",
    "zh": "<think>\n让我们用中文一步一步推理。\n",
}


def _get_lang_from_args(args):
    if not args:
        return "en"
    return getattr(args, "mgsm_lang", None) or getattr(args, "lang", None) or "en"


def get_assistant_think_prefill(args) -> str:
    return _THINK_PREFILLS.get(_get_lang_from_args(args), "")


def _with_language_directive(content: str, lang: str) -> str:
    directive = _LANGUAGE_DIRECTIVES.get(lang)
    if not directive or directive in content:
        return content
    return f"{content.rstrip()}\n\n{directive}"


def _fetch_translation(section: str, role: str, task: str, lang: str):
    try:
        lang_map = _TRANSLATIONS.get(lang, {})
        sec = lang_map.get(section, {})
        # direct role replacement
        if isinstance(sec.get(role), str):
            return sec.get(role)
        # judger special cases
        if role == "judger":
            # try per-task judger template
            jud = sec.get("judger", {})
            if isinstance(jud, dict) and isinstance(jud.get(task), str):
                return jud.get(task)
            if isinstance(sec.get("judger_default"), str):
                return sec.get("judger_default")
        # single-agent default
        if section == "single_agent" and isinstance(sec.get("default"), str):
            return sec.get("default")
        return None
    except Exception:
        return None


def build_agent_message_sequential_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    lang = _get_lang_from_args(args)
    system_message = _TRANSLATIONS.get(lang, {}).get("system_message", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")

    # If a full translation template exists, use it
    tr = _fetch_translation("sequential_latent", role, getattr(args, "task", ""), lang)
    if tr is not None:
        user_prompt = _with_language_directive(tr.format(question=question, context=context), lang)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if role == "planner":
        user_prompt = f"""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: {question}

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:
"""
    
    elif role == "critic":
        user_prompt = f"""
Question: {question}

You are a Critic Agent to evaluate the correctness of the input plan for the given question and provide helpful feedback for improving the plan.
The plan information is provided in latent KV representation format. Review the plan and question and output:
(1) original plan contents
(2) constructive feedback on the original plan.

Format your response as follows:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""
    
    elif role == "refiner":
        user_prompt = f"""
Question: {question}

You are a Refiner Agent to provide a refined step-by-step plan for solving the given question.
You are provided with:
(1) latent-format information: a previous plan with feedback
(2) text-format information: the input question you need to solve.

Based on the input, write a refined and improved plan to solve the question. Make sure your output plan is correct and concise.

Now, output your refined plan below:
"""
    
    elif role == "judger":
        if args.task in ['gsm8k', 'mgsm', 'aime2024', 'aime2025']:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        
        elif args.task in ["arc_easy", "arc_challenge", "gpqa", 'medqa']:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif args.task in ["mbppplus", "humanevalplus"]:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve.

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block.

Now, reason step by step and output the final answer inside ```python
YOUR_PYTHON_CODE
```.
"""

        elif args.task in ["winogrande"]:
            user_prompt = f"""
Target Question: {question}

You are a helpful assistant. You are provided with latent information for reference and a target question to solve. 

The latent information might contain irrelevant contents. Ignore it if it is not helpful for solving the target question.

You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        else: 
            raise NotImplementedError(f"Task {args.task} not implemented in v5 judger prompt.")
        
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": _with_language_directive(user_prompt, lang)},
    ]


def build_agent_message_hierarchical_latent_mas(role: str, question: str, context: str = "", method=None, args=None):

    lang = _get_lang_from_args(args)
    system_message = _TRANSLATIONS.get(lang, {}).get("system_message", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")

    tr = _fetch_translation("hierarchical_latent", role, getattr(args, "task", ""), lang)
    if tr is not None:
        user_content = _with_language_directive(tr.format(question=question, context=context), lang)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    assert method in ["latent_mas"], "this prompt only for latent_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    if args.task in ['gsm8k', 'mgsm', 'aime2024', 'aime2025']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:

        if args.task == "medqa":

            if role == "planner":
                user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""
            elif role == "critic":
                user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}     

Your response:
"""
            elif role == "refiner":
                user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:       
"""
            elif role == "judger":

                user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. 

Input Question: {question}

Your response:
"""

        else:
            if role == "planner":
                user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""
    
            elif role == "critic":
                user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}     

Your response:
"""
    
            elif role == "refiner":
                user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:       
"""
            elif role == "judger":

                user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Input Question: {question}

Your response:
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import math
def add(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step by step: please provide an efficient and self-contained Python function that solves the following problem in a markdown code block:\n```\nYOUR_PYTHON_CODE\n```.
You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 
    
Input Question: {question}

Your response:
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": _with_language_directive(user_content, lang)},
    ]


def build_agent_messages_sequential_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["text_mas"], "only for text_mas method"
    assert "qwen" in args.model_name.lower(), "only for qwen models"

    lang = _get_lang_from_args(args)
    system_message = _TRANSLATIONS.get(lang, {}).get("system_message", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")

    tr = _fetch_translation("text_mas_sequential", role, getattr(args, "task", ""), lang)
    if tr is not None:
        user_content = _with_language_directive(tr.format(question=question, context=context), lang)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    # truncate context if needed
    ctx = context[: args.text_mas_context_length]

    if role == "planner":
        user_content = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""

    elif role == "critic":
        user_content = f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{question}

## Plan from Planner Agent:
{ctx}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""

    elif role == "refiner":
        user_content = f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{question}

## Original Plan and Critic Feedback:
{ctx}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:
"""

    elif role == "judger":
        task = getattr(args, "task", None)

        if task in ["gsm8k", "aime2024", "aime2025"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["mbppplus", "humanevalplus"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
"""
            
        elif task in ["winogrande"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        else:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and present your final answer clearly at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": _with_language_directive(user_content, lang)},
    ]


def build_agent_messages_hierarchical_text_mas(role: str, question: str, context: str = "", method=None, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    assert method in ["text_mas"], "this prompt only for text_mas method"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"
    
    lang = _get_lang_from_args(args)
    system_message = _TRANSLATIONS.get(lang, {}).get("system_message", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")

    tr = _fetch_translation("text_mas_hierarchical", role, getattr(args, "task", ""), lang)
    if tr is not None:
        user_content = _with_language_directive(tr.format(question=question, context=context), lang)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    if args.task in ['gsm8k', 'mgsm', 'aime2024', 'aime2025']:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Input Question: {question}

Your response:       
"""
        elif role == "judger":

            user_content = f"""
You are a task summarizer. Given the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["mbppplus", "humanevalplus"]:
        
        if role == "planner":
            user_content = f"""
You are a math agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "critic":
            user_content = f"""
You are a science agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "refiner":
            user_content = f"""
You are a code agent. You must put all python code as self-contained Python function in markdown code blocks. For example ```python
import needed_library
def FUNC_NAME(a, b):
    return a + b```. Do not add any other contents inside the markdown code block. 

Input Question: {question}

Your response:
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the final answer in markdown python code block.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

Input Question: {question}

Your response:
"""

    elif args.task in ["winogrande"]:
        if role == "planner":
            user_content = f"""
You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""
    
        elif role == "critic":
            user_content = f"""
You are a science agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}     

Your response:
"""
    
        elif role == "refiner":
            user_content = f"""
You are a code agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:       
"""
        elif role == "judger":
            user_content = f"""
You are a task summarizer. Given the input question and responses from previous agents as reference, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.

Content from Previous Agent:
{context[:args.text_mas_context_length]}

"Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box."

Input Question: {question}

Your response:
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": _with_language_directive(user_content, lang)},
    ]


def build_agent_messages_single_agent(question: str, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert args.method in ["baseline"], "this prompt only for baseline method (single agent)"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    lang = _get_lang_from_args(args)
    system_message = _TRANSLATIONS.get(lang, {}).get("system_message", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")

    tr = _fetch_translation("single_agent", "default", getattr(args, "task", ""), lang)
    if tr is not None:
        user_content = _with_language_directive(tr.format(question=question), lang)
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

    task = args.task

    if task in ["gsm8k", "mgsm", "aime2024", "aime2025"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": _with_language_directive(user_content, lang)},
    ]
