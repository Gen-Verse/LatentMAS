"""Optional training / fine-tuning utilities.

This subpackage is fully optional and import-guarded: importing it (or the rest of
the repo) does not require Unsloth to be installed. The heavy dependencies are only
imported when a trainer is actually constructed/run, so the inference pipeline keeps
working without them.

Install the optional extra with::

    pip install -e .[unsloth]
"""

__author__ = "Lineesha Kamana, Himon Thakur"
__copyright__ = "Copyright 2026, Lineesha Kamana, Himon Thakur"
__credits__ = ["Lineesha Kamana", "Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Lineesha Kamana"
__email__ = "lpk5305@psu.edu, hthakur@uccs.edu"
__status__ = "prototype"

__all__ = ["UnslothTrainer"]


def __getattr__(name):
    # Lazy attribute access so `from training import UnslothTrainer` does not import
    # Unsloth at package-import time.
    if name == "UnslothTrainer":
        from .unsloth_trainer import UnslothTrainer

        return UnslothTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
