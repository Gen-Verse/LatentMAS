import sys
from pathlib import Path

# Sibling repos on this machine (LRL-MRRE-MAS, brain, logit-distill, ...) register
# editable installs that inject their own src/ onto sys.path and shadow this repo's
# same-named packages (shared, latent_coordination, mrre_drift) via PathFinder, which
# resolves regular (non-namespace) packages on first match. Force this repo's src/ to
# be checked first so tests import our code, not a sibling repo's same-named module.
_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)
