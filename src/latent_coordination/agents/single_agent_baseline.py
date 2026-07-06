import logging

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)

class SingleAgentOneFlow:
    """Adversarial evaluation script: tests the strongest single-model backbone in a 
    multi-turn, KV-cache-preserving role-play setup to establish a OneFlow baseline.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def run_roleplay(self, prompt: str) -> str:
        logger.info("Running single-agent adversarial evaluation (OneFlow Baseline).")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
            
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=512, 
            use_cache=True,  # KV-cache-preserving
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
