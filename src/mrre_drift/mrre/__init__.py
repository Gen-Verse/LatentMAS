from mrre_drift.mrre.stage1 import CrossLingualEnhancer
from mrre_drift.mrre.stage2 import TargetLanguageAnchorer
from mrre_drift.mrre.vectors import EnhancementVectors, PromptPair, compute_enhancement_vectors
from mrre_drift.mrre.anchoring import AnchoringVectors, LanguageForcingPair, compute_anchoring_vectors
from mrre_drift.mrre.hooks import register_injection_hooks

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = [
    "CrossLingualEnhancer",
    "TargetLanguageAnchorer",
    "EnhancementVectors",
    "AnchoringVectors",
    "PromptPair",
    "LanguageForcingPair",
    "compute_enhancement_vectors",
    "compute_anchoring_vectors",
    "register_injection_hooks",
]
