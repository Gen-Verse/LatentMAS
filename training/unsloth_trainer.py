"""Unsloth-based LoRA fine-tuning trainer (optional).

Unsloth (https://github.com/unslothai/unsloth) provides fast, memory-efficient
fine-tuning. All heavy imports happen inside methods so the rest of the repo never
needs Unsloth installed unless training is actually run.

Typical usage (see ``train.py`` and ``configs/unsloth_train.yaml``)::

    from training import UnslothTrainer
    trainer = UnslothTrainer(config_dict)
    trainer.run()
"""

from typing import Any, Dict, Optional

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"



_DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _require_unsloth():
    try:
        # Importing unsloth first is recommended for its patching to take effect.
        from unsloth import FastLanguageModel  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "Unsloth is not installed. Install the optional extra with "
            "`pip install -e .[unsloth]` (or follow https://github.com/unslothai/unsloth#installation)."
        ) from exc
    return FastLanguageModel


class UnslothTrainer:
    """Thin, config-driven wrapper around Unsloth + TRL's ``SFTTrainer``.

    The config is a nested dict with ``model``, ``lora``, ``data``, ``training`` and
    ``save`` sections. See ``configs/unsloth_train.yaml`` for a documented example.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.model_cfg = self.config.get("model", {})
        self.lora_cfg = self.config.get("lora", {})
        self.data_cfg = self.config.get("data", {})
        self.train_cfg = self.config.get("training", {})
        self.save_cfg = self.config.get("save", {})

        self.model = None
        self.tokenizer = None
        self.trainer = None

    # ------------------------------------------------------------------ model
    def load_model(self):
        FastLanguageModel = _require_unsloth()

        model_name = self.model_cfg.get("model_name")
        if not model_name:
            raise ValueError("config.model.model_name is required.")
        max_seq_length = int(self.model_cfg.get("max_seq_length", 2048))

        print(f"[Unsloth] Loading base model: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=self.model_cfg.get("dtype", None),
            load_in_4bit=bool(self.model_cfg.get("load_in_4bit", True)),
        )

        print("[Unsloth] Attaching LoRA adapters")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=int(self.lora_cfg.get("r", 16)),
            target_modules=self.lora_cfg.get("target_modules", _DEFAULT_TARGET_MODULES),
            lora_alpha=int(self.lora_cfg.get("lora_alpha", 16)),
            lora_dropout=float(self.lora_cfg.get("lora_dropout", 0.0)),
            bias=self.lora_cfg.get("bias", "none"),
            use_gradient_checkpointing=self.lora_cfg.get("use_gradient_checkpointing", "unsloth"),
            random_state=int(self.lora_cfg.get("random_state", 42)),
            use_rslora=bool(self.lora_cfg.get("use_rslora", False)),
        )
        return self.model, self.tokenizer

    # ------------------------------------------------------------------- data
    def build_dataset(self):
        from datasets import load_dataset

        dataset_name = self.data_cfg.get("dataset_name")
        if not dataset_name:
            raise ValueError("config.data.dataset_name is required (HF id or local json/jsonl path).")
        split = self.data_cfg.get("dataset_split", "train")

        if dataset_name.endswith((".json", ".jsonl")):
            dataset = load_dataset("json", data_files=dataset_name, split=split)
        else:
            config_name = self.data_cfg.get("dataset_config")
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)

        text_field = self.data_cfg.get("text_field", "text")
        messages_field = self.data_cfg.get("messages_field")

        # If the dataset stores chat messages, render them with the tokenizer's
        # chat template into the target text field.
        if messages_field:
            def _format(example):
                rendered = self.tokenizer.apply_chat_template(
                    example[messages_field], tokenize=False, add_generation_prompt=False
                )
                return {text_field: rendered}

            dataset = dataset.map(_format)

        self.data_cfg["_resolved_text_field"] = text_field
        return dataset

    # --------------------------------------------------------------- training
    def build_trainer(self, dataset):
        from transformers import TrainingArguments
        from trl import SFTTrainer

        text_field = self.data_cfg.get("_resolved_text_field", self.data_cfg.get("text_field", "text"))
        max_seq_length = int(self.model_cfg.get("max_seq_length", 2048))

        ta_kwargs = dict(
            output_dir=self.train_cfg.get("output_dir", "outputs/unsloth"),
            per_device_train_batch_size=int(self.train_cfg.get("per_device_train_batch_size", 2)),
            gradient_accumulation_steps=int(self.train_cfg.get("gradient_accumulation_steps", 4)),
            warmup_steps=int(self.train_cfg.get("warmup_steps", 5)),
            learning_rate=float(self.train_cfg.get("learning_rate", 2e-4)),
            logging_steps=int(self.train_cfg.get("logging_steps", 1)),
            optim=self.train_cfg.get("optim", "adamw_8bit"),
            weight_decay=float(self.train_cfg.get("weight_decay", 0.01)),
            lr_scheduler_type=self.train_cfg.get("lr_scheduler_type", "linear"),
            seed=int(self.train_cfg.get("seed", 42)),
            report_to=self.train_cfg.get("report_to", "none"),
        )
        # max_steps takes precedence over num_train_epochs when > 0.
        max_steps = int(self.train_cfg.get("max_steps", -1))
        if max_steps and max_steps > 0:
            ta_kwargs["max_steps"] = max_steps
        else:
            ta_kwargs["num_train_epochs"] = float(self.train_cfg.get("num_train_epochs", 1))

        args = TrainingArguments(**ta_kwargs)

        # TRL changed SFTTrainer's signature across versions: older releases accept
        # `tokenizer`, `dataset_text_field` and `max_seq_length` directly, newer ones
        # expect them via SFTConfig. Try the classic signature, then fall back.
        try:
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                dataset_text_field=text_field,
                max_seq_length=max_seq_length,
                args=args,
                packing=bool(self.train_cfg.get("packing", False)),
            )
        except TypeError:
            from trl import SFTConfig

            sft_args = SFTConfig(
                dataset_text_field=text_field,
                max_seq_length=max_seq_length,
                packing=bool(self.train_cfg.get("packing", False)),
                **ta_kwargs,
            )
            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=dataset,
                args=sft_args,
            )
        return self.trainer

    def train(self):
        if self.trainer is None:
            raise RuntimeError("Call build_trainer(...) before train().")
        return self.trainer.train()

    # ------------------------------------------------------------------- save
    def save(self):
        save_method = self.save_cfg.get("save_method", "lora")
        output_dir = self.save_cfg.get("output_dir", "outputs/unsloth-final")

        if save_method == "lora":
            print(f"[Unsloth] Saving LoRA adapters to {output_dir}")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        elif save_method in ("merged_16bit", "merged_4bit"):
            print(f"[Unsloth] Saving merged model ({save_method}) to {output_dir}")
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method=save_method)
        else:
            raise ValueError(f"Unknown save_method: {save_method!r}")

        # Optional GGUF export so the fine-tuned model can be served via the
        # llama.cpp backend (--use_llamacpp --llamacpp_model_path ...).
        if self.save_cfg.get("gguf", False):
            gguf_quant = self.save_cfg.get("gguf_quant", "q4_k_m")
            gguf_dir = self.save_cfg.get("gguf_output_dir", output_dir)
            print(f"[Unsloth] Exporting GGUF ({gguf_quant}) to {gguf_dir}")
            self.model.save_pretrained_gguf(
                gguf_dir, self.tokenizer, quantization_method=gguf_quant
            )

    # -------------------------------------------------------------------- run
    def run(self):
        """End-to-end: load model, build data, train, save."""
        self.load_model()
        dataset = self.build_dataset()
        self.build_trainer(dataset)
        self.train()
        self.save()
        print("[Unsloth] Done.")
