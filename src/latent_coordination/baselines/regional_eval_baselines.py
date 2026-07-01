import logging

logger = logging.getLogger(__name__)

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


class RegionalLLMBaseline:
    """Generic text generation baseline for regional benchmarks."""
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", **kwargs):
        logger.info(f"Initializing {self.__class__.__name__} with {model_name}.")
        try:
            from transformers import pipeline
            self.pipeline = pipeline("text-generation", model=model_name, device_map="auto")
        except ImportError:
            self.pipeline = None
            logger.warning(f"transformers not installed. {self.__class__.__name__} will fail.")

    def __call__(self, prompt: str, *args, **kwargs):
        if not self.pipeline:
            raise ImportError("transformers pipeline unavailable.")
        result = self.pipeline(prompt, max_new_tokens=128)
        return result[0]['generated_text']

class SeaHelmBaseline(RegionalLLMBaseline):
    """Baseline evaluating models against SEA-HELM regional tasks."""

class SeaCrowdBaseline(RegionalLLMBaseline):
    """Baseline for SEACrowd multilingual reasoning."""

class SeaEvalBaseline(RegionalLLMBaseline):
    """Baseline for SEAEval reasoning capabilities."""

class SeaLionBaseline(RegionalLLMBaseline):
    """Baseline adapter for SEALion family models."""
    def __init__(self, model_name="aisingapore/sea-lion-7b-instruct", **kwargs):
        super().__init__(model_name=model_name, **kwargs)

class SeaExamBaseline(RegionalLLMBaseline):
    """Baseline for SEAExam regional logic and QA."""

class SeaBenchBaseline(RegionalLLMBaseline):
    """Baseline for SeaBench suite."""

class MultiChallengeBaseline(RegionalLLMBaseline):
    """Baseline for MultiChallenge benchmark."""

class XQuadBaseline:
    """Baseline for XQuAD cross-lingual question answering."""
    def __init__(self, model_name="deepset/xlm-roberta-large-squad2", **kwargs):
        logger.info(f"Initializing XQuAD baseline with {model_name}.")
        try:
            from transformers import pipeline
            self.qa_pipeline = pipeline("question-answering", model=model_name, device_map="auto")
        except ImportError:
            self.qa_pipeline = None
            logger.warning("transformers not installed. XQuadBaseline will fail on call.")

    def __call__(self, context: str, question: str, *args, **kwargs):
        if not self.qa_pipeline:
            raise ImportError("transformers pipeline unavailable.")
        result = self.qa_pipeline(question=question, context=context)
        return result.get("answer", "")

class MlqaBaseline:
    """Baseline for MLQA (Multilingual Question Answering)."""
    def __init__(self, model_name="deepset/xlm-roberta-large-squad2", **kwargs):
        logger.info(f"Initializing MLQA baseline with {model_name}.")
        try:
            from transformers import pipeline
            self.qa_pipeline = pipeline("question-answering", model=model_name, device_map="auto")
        except ImportError:
            self.qa_pipeline = None
            logger.warning("transformers not installed. MlqaBaseline will fail on call.")

    def __call__(self, context: str, question: str, *args, **kwargs):
        if not self.qa_pipeline:
            raise ImportError("transformers pipeline unavailable.")
        result = self.qa_pipeline(question=question, context=context)
        return result.get("answer", "")
