"""
Surgical Multilingual Reasoning via Representation Engineering (MRRE).

Two-stage cross-lingual activation steering for low-resource languages:
  Stage 1 (CrossLingualEnhancer)   — injects en→tgt mean-diff vectors at
                                     intermediate layers to unlock English
                                     reasoning circuits for LRL inputs.
  Stage 2 (TargetLanguageAnchorer) — injects tgt-forcing vectors at final
                                     layers to pull output back to the target
                                     language after Stage 1 steering.

Surgical variant (SurgicalMRRE) uses CollapseProfile (Logit Lens + CRAF) to
pick exactly the layers that need intervention, with a configurable exclusion
zone for vision-fusion layers in VLMs.

§Methodology
Given K paired prompts (xₖᵉⁿ, xₖᵗᵍᵗ), collect mean-pooled hidden states Hₑₙ⁽ˡ⁾, H_tgt⁽ˡ⁾ ∈ ℝ^{K×d} at layer ℓ. Define the contrastive matrix C⁽ˡ⁾ = H_tgt⁽ˡ⁾ − Hₑₙ⁽ˡ⁾ with truncated SVD C⁽ˡ⁾ = UΣVᵀ, V = [v₁,…,v_d]. The language subspace is U_L = span(v₁,…,v_r) (r at the cumulative-energy knee) and the reasoning subspace U_R is its orthogonal complement.

§4.1
Proposition (CLAP is the rank-1 scalar reduction of the language subspace). Let H⁽ˡ⁾ = [Hₑₙ⁽ˡ⁾; H_tgt⁽ˡ⁾] be the global-mean-centred stack with leading singular direction u₁. Because the dominant between-group variance in H⁽ˡ⁾ is the centroid difference μ_tgt − μₑₙ = (1/K)·𝟙ᵀC⁽ˡ⁾, the dominant axis u₁ coincides (up to sign) with the leading contrastive direction v₁ of C⁽ˡ⁾ whenever the cross-lingual difference is the leading source of variance — exactly the regime indicated by the sharp elbow in the C⁽ˡ⁾ spectrum. Thus CLAP's δ⁽ˡ⁾ = c̄os(Hₑₙ, u₁) − c̄os(H_tgt, u₁) is the signed scalar projection gap along v₁, i.e. the rank-1 summary of the same subspace U_L = span(v₁,…,v_r) used for projection-based steering. CLAP localizes where the language axis separates EN/tgt (a per-layer scalar); the subspace projection acts on it (a rank-r operator).

§Discussion
Stage 1 (enhancement) adds v_enh = mean(hₑₙ − h_tgt) ≈ +v₁ in the mid-stack reasoning band — moving the target representation toward the English language frame so the English-tuned reasoning circuitry engages, without touching U_R.
Stage 2 (anchoring) adds v_anc = mean(h_tgt − hₑₙ) ≈ −v₁ in the late commitment band — moving the language frame back to the target so the unembedding decodes target script.
The full pipeline is therefore a depth-dependent trajectory through U_L (toward English mid-stack, back to target late) over an invariant reasoning subspace U_R. The ICLR projection h′ = h − λ·P_L·h is the pure limit that strips the language frame entirely to isolate U_R; the AAAI anchoring is its late-layer partial inverse.

Anchoring re-injects the U_L component characterised by the companion's decomposition.
A companion submission studies the complementary reasoning axis of the same decomposition.
"""

from mrre_drift.mrre.surgical import SurgicalMRRE, SurgicalMRREConfig
from mrre_drift.mrre.stage1 import CrossLingualEnhancer
from mrre_drift.mrre.stage2 import TargetLanguageAnchorer
from mrre_drift.interpret.collapse import CollapseDetector, CollapseProfile

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

try:
    from mrre_drift.interpret.logit_lens import (
        LogitLens,
        LogitLensScan,
        LayerLensResult,
        get_lm_head_components,
    )
    from mrre_drift.interpret.craf import CRAF, CRAFProfile, LayerCRAFResult, ConceptDirection
except ImportError:
    pass

__all__ = [
    "SurgicalMRRE",
    "SurgicalMRREConfig",
    "CrossLingualEnhancer",
    "TargetLanguageAnchorer",
    "CollapseDetector",
    "CollapseProfile",
    "LogitLens",
    "LogitLensScan",
    "LayerLensResult",
    "get_lm_head_components",
    "CRAF",
    "CRAFProfile",
    "LayerCRAFResult",
    "ConceptDirection",
]
