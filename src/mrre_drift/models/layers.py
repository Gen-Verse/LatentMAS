"""
Thin re-export from shared.model_utils.

All three projects use the same layer-resolution logic; the canonical
implementation lives in shared.model_utils to avoid drift.
"""

from shared.model_utils import get_transformer_layers, layer_ids_from_fractions  # noqa: F401

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.1.0"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

__all__ = ["get_transformer_layers", "layer_ids_from_fractions"]
