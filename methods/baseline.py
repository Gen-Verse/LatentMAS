from typing import Dict, List

from models import ModelWrapper
from prompts import build_agent_messages_single_agent
from utils import extract_gsm8k_answer, normalize_answer, extract_markdown_python_block, run_with_timeout


class BaselineMethod:
    def __init__(
            self,
            model: ModelWrapper,
            *,
            max_new_tokens: int = 256,
            temperature: float = 0.7,
            top_p: float = 0.95,
            generate_bs: int = 1,
            use_vllm: bool = False,
            args=None,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.method_name = "baseline"
        self.args = args
        self.task = args.task

    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_messages = [
            build_agent_messages_single_agent(question=item["question"], args=self.args)
            for item in items
        ]

        prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )

        if self.use_vllm:
            generated_batch = self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            generated_batch, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        results: List[Dict] = []

        for idx, item in enumerate(items):
            generated_text = generated_batch[idx]

            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(generated_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(generated_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(generated_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            mask = attention_mask[idx].bool()
            trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
            agent_trace = {
                "name": "SingleAgent",
                "role": "singleagent",
                "input": prompts[idx],
                "input_ids": trimmed_ids,
                "input_tokens": tokens_batch[idx],
                "output": generated_text,
            }
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": generated_text,
                    "agents": [agent_trace],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]