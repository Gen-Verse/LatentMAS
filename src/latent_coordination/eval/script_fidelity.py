"""Compatibility shim: SFR/IFL and Language-Consistency evaluators moved to shared.

The implementation now lives in ``src/shared/script_fidelity.py`` (strategy.md
§3.5: IFL/SFR is a *shared metric* consumed by both ``latent_coordination`` and
``mechanistic_disentangle``, so it cannot live inside either firewalled
package). This module re-exports everything so existing latent-side imports
keep working.
"""

from shared.script_fidelity import *  # noqa: F401,F403
from shared.script_fidelity import (  # noqa: F401 — explicit for static analysis
    SCRIPT_UNICODE_RANGES,
    ScriptFidelityEvaluator,
    LanguageConsistencyEvaluator,
)
