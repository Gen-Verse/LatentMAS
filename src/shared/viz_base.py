"""
Shared visualisation base utilities: style setup, figure saving, colour palettes.

All plot modules in mechanistic_disentangle.viz and latent_coordination.viz import from here
to maintain consistent aesthetics across all paper figures.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language colour palette (ISO-639-1 codes -> hex colour)
# ---------------------------------------------------------------------------

LANGUAGE_COLOR_PALETTE: Dict[str, str] = {
    "en": "#4878CF",   # blue
    "th": "#D65F5F",   # red
    "my": "#6ACC65",   # green
    "km": "#B47CC7",   # purple
    "lo": "#C4AD66",   # gold
    "vi": "#77BEDB",   # sky blue
    "id": "#F0A500",   # amber
    "ms": "#E07B54",   # coral
    "jv": "#8EBF72",   # sage
    "su": "#5FA8AD",   # teal
    "fil": "#E8A0BF",  # pink
    "am": "#A0522D",   # sienna
    "sw": "#708090",   # slate
    "zh": "#DC143C",   # crimson
}

_DEFAULT_COLOR = "#888888"


def get_language_color(lang: str) -> str:
    """Return a colour hex string for a given language ISO code."""
    return LANGUAGE_COLOR_PALETTE.get(lang.lower(), _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class VizConfig:
    """Global visualisation configuration.

    Attributes:
        fig_width: Default figure width in inches.
        fig_height: Default figure height in inches.
        dpi: Dots per inch for raster exports.
        font_size: Base font size (axis labels, ticks).
        title_size: Title font size.
        style: Seaborn style name.
        context: Seaborn context name.
        palette: Colour palette name or list.
        save_pdf: Whether to also save a PDF alongside the PNG.
        tight_layout: Whether to apply tight_layout before saving.
    """

    fig_width: float = 10.0
    fig_height: float = 5.0
    dpi: int = 150
    font_size: int = 11
    title_size: int = 13
    style: str = "whitegrid"
    context: str = "paper"
    palette: str = "mako"
    save_pdf: bool = True
    tight_layout: bool = True

    @property
    def figsize_default(self) -> tuple:
        return (self.fig_width, self.fig_height)


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------

def setup_style(cfg: Optional[VizConfig] = None) -> VizConfig:
    """Apply seaborn/matplotlib style settings and return the active VizConfig.

    Safe to call multiple times; no-ops if matplotlib/seaborn are unavailable.

    Args:
        cfg: Optional VizConfig; defaults to VizConfig() if None.

    Returns:
        The active VizConfig (useful for chaining).
    """
    if cfg is None:
        cfg = VizConfig()

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for server environments
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style=cfg.style, context=cfg.context, font_scale=cfg.font_size / 10.0)
        plt.rcParams.update({
            "figure.dpi": cfg.dpi,
            "axes.titlesize": cfg.title_size,
            "axes.labelsize": cfg.font_size,
            "xtick.labelsize": cfg.font_size - 1,
            "ytick.labelsize": cfg.font_size - 1,
            "legend.fontsize": cfg.font_size - 1,
            "figure.figsize": (cfg.fig_width, cfg.fig_height),
        })
    except ImportError:
        logger.warning("matplotlib/seaborn not available; skipping style setup.")

    return cfg


# ---------------------------------------------------------------------------
# Figure persistence
# ---------------------------------------------------------------------------

def save_figure(fig, path: Path | str, cfg: Optional[VizConfig] = None) -> None:
    """Save a matplotlib figure to disk as PNG (and optionally PDF).

    Creates parent directories as needed.  Closes the figure after saving
    to avoid memory leaks in long-running pipeline stages.

    Args:
        fig: matplotlib Figure object.
        path: Destination file path (with or without extension).
        cfg: VizConfig controlling DPI and PDF saving.  Defaults to VizConfig().
    """
    if cfg is None:
        cfg = VizConfig()

    try:
        import matplotlib.pyplot as plt

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        png_path = path.with_suffix(".png")
        if cfg.tight_layout:
            try:
                fig.tight_layout()
            except Exception:
                pass  # some figure layouts don't support tight_layout

        fig.savefig(png_path, dpi=cfg.dpi, bbox_inches="tight")
        logger.debug("Saved figure: %s", png_path)

        if cfg.save_pdf:
            pdf_path = path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            logger.debug("Saved figure: %s", pdf_path)

        plt.close(fig)

    except ImportError:
        logger.warning("matplotlib not available; figure not saved.")
    except Exception as exc:
        logger.error("Failed to save figure to %s: %s", path, exc)
