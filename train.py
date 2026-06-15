"""CLI entrypoint for optional Unsloth fine-tuning.

Example:
    python train.py --config configs/unsloth_train.yaml

Any ``section.key`` may be overridden from the command line, e.g.:
    python train.py --config configs/unsloth_train.yaml \
        --set model.model_name=Qwen/Qwen3-4B training.max_steps=120
"""

import argparse

from utils import load_yaml_config, set_seed

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Lineesha Kamana"
__email__ = "lpk5305@psu.edu, hthakur@uccs.edu"
__status__ = "prototype"


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _apply_overrides(config: dict, overrides) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item!r}")
        dotted, value = item.split("=", 1)
        keys = dotted.split(".")
        node = config
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _coerce(value)
    return config


def main():
    parser = argparse.ArgumentParser(description="Unsloth LoRA fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML training config")
    parser.add_argument("--set", nargs="*", dest="overrides", default=[],
                        help="Override config values, e.g. training.max_steps=120")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    config = _apply_overrides(config, args.overrides)

    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)

    # Imported here so `python run.py ...` never pulls in Unsloth.
    from training import UnslothTrainer

    UnslothTrainer(config).run()


if __name__ == "__main__":
    main()
