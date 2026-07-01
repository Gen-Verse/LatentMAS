import logging
import torch
from typing import List

logger = logging.getLogger(__name__)

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """chrF (Character n-gram F-score) for evaluating text generation and translation."""
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF()
        return chrf.corpus_score(predictions, [references]).score
    except ImportError as exc:
        raise ImportError("chrF evaluation requires the sacrebleu package. Install with: pip install sacrebleu") from exc


def compute_comet(predictions: List[str], references: List[str], sources: List[str]) -> float:
    """COMET metric for translation and high-quality generation evaluation."""
    logger.info("Computing COMET scores for evaluation.")
    try:
        from comet import download_model, load_from_checkpoint
        # Uses standard wmt22-comet-da as the default COMET checkpoint
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(sources, predictions, references)]
        model_output = model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        return float(model_output.system_score)
    except ImportError as exc:
        raise ImportError(
            "COMET requires the unbabel-comet package and model weights. "
            "Install with: pip install unbabel-comet"
        ) from exc


# Regex patterns for Unicode blocks to measure script fidelity
SCRIPT_BLOCKS = {
    "th": r"[\u0E00-\u0E7F]", # Thai
    "lo": r"[\u0E80-\u0EFF]", # Lao
    "km": r"[\u1780-\u17FF]", # Khmer
    "my": r"[\u1000-\u109F]", # Burmese
    "am": r"[\u1200-\u137F]", # Amharic
    "bn": r"[\u0980-\u09FF]", # Bengali
    "te": r"[\u0C00-\u0C7F]", # Telugu
}

def _detect_script_ratio(text: str, target_lang: str) -> float:
    """Calculates the ratio of tokens containing the target script."""
    if not text:
        return 0.0
    pattern = SCRIPT_BLOCKS.get(target_lang)
    if not pattern:
        # Fallback if no specific script block is registered for the lang
        return 1.0
        
    tokens = text.split()
    if not tokens:
        return 0.0
        
    import re
    match_count = sum(1 for token in tokens if re.search(pattern, token))
    return match_count / len(tokens)


def compute_sfr(predictions: List[str], target_lang: str = "th") -> float:
    """Script Fidelity Rate (SFR) metric for evaluating character/script consistency.
    Formula: SFR = (1/M) * sum(I(script(t_m) == script_target))
    """
    logger.info(f"Computing Script Fidelity Rate (SFR) for target lang: {target_lang}.")
    if not predictions:
        return 0.0
    ratios = [_detect_script_ratio(p, target_lang) for p in predictions]
    return sum(ratios) / len(ratios)


def compute_ifl(predictions: List[str], target_lang: str = "th") -> float:
    """Involuntary Fidelity Loss (IFL) metric from typical NLP metrics.
    Formula: IFL = 1.0 - SFR
    """
    logger.info("Computing Involuntary Fidelity Loss (IFL) metric.")
    sfr = compute_sfr(predictions, target_lang)
    return 1.0 - sfr


def compute_cosine_similarity(tensor_a: torch.Tensor, tensor_b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Standard cosine similarity between latent representations."""
    return torch.nn.functional.cosine_similarity(tensor_a, tensor_b, dim=dim)


def compute_clap(source_features: torch.Tensor, target_features: torch.Tensor) -> float:
    """Cross-Lingual Alignment Probe (CLAP) metric from the MRRE drift paper.
    Calculates the alignment delta: mean_cos(source, u1) - mean_cos(target, u1)
    """
    logger.info("Computing Cross-Lingual Alignment Probe (CLAP).")
    if source_features.size(0) == 0 or target_features.size(0) == 0:
        return 0.0

    # Combined matrix
    combined = torch.cat([source_features, target_features], dim=0).float()
    combined = combined - combined.mean(dim=0, keepdim=True)
    
    try:
        # SVD
        U, S, Vh = torch.linalg.svd(combined, full_matrices=False)
        u1 = Vh[0]
    except Exception:
        # Fallback to mean difference direction
        diff = (source_features.mean(0) - target_features.mean(0)).float()
        u1 = diff / diff.norm().clamp(min=1e-8)
        
    u1 = u1 / u1.norm().clamp(min=1e-8)
    
    # Orient towards source centroid (e.g., English)
    diff_mean = (source_features.mean(dim=0) - target_features.mean(dim=0)).float().to(u1.device)
    if (u1 @ diff_mean) < 0:
        u1 = -u1
        
    def _mean_cos(matrix: torch.Tensor, d: torch.Tensor):
        norms = matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
        d_norm = d / d.norm().clamp(min=1e-8)
        sims = (matrix / norms) @ d_norm
        return float(sims.mean())

    source_cos = _mean_cos(source_features.float(), u1)
    target_cos = _mean_cos(target_features.float(), u1)
    
    return source_cos - target_cos


def compute_drift(baseline_activations: torch.Tensor, current_activations: torch.Tensor) -> float:
    """Representation drift metric (used in MRRE-Drift bounds)."""
    return torch.mean(torch.abs(baseline_activations - current_activations)).item()


def compute_cka_alignment(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Centered Kernel Alignment (CKA) between two latent feature spaces."""
    logger.info("Computing linear CKA alignment.")
    
    # Normalize features
    x = features_x - features_x.mean(dim=0, keepdim=True)
    y = features_y - features_y.mean(dim=0, keepdim=True)
    
    # Gram matrices
    gram_x = torch.mm(x, x.t())
    gram_y = torch.mm(y, y.t())
    
    # Frobenius dot product
    dot_xy = torch.sum(gram_x * gram_y)
    norm_x = torch.sqrt(torch.sum(gram_x * gram_x))
    norm_y = torch.sqrt(torch.sum(gram_y * gram_y))
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
        
    cka = dot_xy / (norm_x * norm_y)
    return cka.item()


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Exact Match (EM) metric."""
    if not predictions or not references or len(predictions) != len(references):
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if str(p).strip() == str(r).strip())
    return float(matches) / len(predictions)


def compute_chrf_plus(predictions: List[str], references: List[str]) -> float:
    """chrF+ (Character n-gram F-score + word n-grams) for translation evaluation."""
    try:
        from sacrebleu.metrics import CHRF
        # Word-order enabled (chrF+ typically uses word_order=2)
        chrf = CHRF(word_order=2)
        return chrf.corpus_score(predictions, [references]).score
    except ImportError as exc:
        raise ImportError("chrF+ evaluation requires the sacrebleu package. Install with: pip install sacrebleu") from exc
