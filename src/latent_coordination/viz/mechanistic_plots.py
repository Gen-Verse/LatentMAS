"""Mechanistic plots for evaluating attention, weight matrix SVDs, and logit-lens evolutions."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch import Tensor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from shared.viz_base import VizConfig, setup_style, save_figure, LANGUAGE_COLOR_PALETTE, get_language_color

__author__ = "Himon Thakur"
__copyright__ = "Copyright 2026, Himon Thakur"
__credits__ = ["Himon Thakur"]
__license__ = "Apache 2.0"
__version__ = "0.0.1"
__maintainer__ = "Himon Thakur"
__email__ = "hthakur@uccs.edu"
__status__ = "prototype"

logger = logging.getLogger(__name__)


def to_numpy(x: Any) -> np.ndarray:
    """Safely convert any tensor or array-like to NumPy ndarray."""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


class MechanisticPlotter:
    """Generates detailed plots for mechanistic analyses (e.g. attention, weights, logit lens)."""

    def __init__(self, config: Optional[VizConfig] = None) -> None:
        self.config = config or VizConfig()
        setup_style(self.config)

    def plot_attention_patterns(
        self,
        attention_weights: Tensor | np.ndarray,
        tokens: List[str],
        layer_id: int,
        head_id: int,
        save_path: Path | str,
    ) -> None:
        """Plot a heatmap of attention weights between tokens."""
        fig, ax = plt.subplots(figsize=(8, 7))

        weights = to_numpy(attention_weights)
        sns.heatmap(
            weights,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Blues",
            ax=ax,
            cbar_kws={"label": "Attention Probability"},
        )

        ax.set_title(f"Attention Pattern: Layer {layer_id}, Head {head_id}")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved attention heatmap to %s", save_path)

    def plot_layer_wise_accuracy(
        self,
        accuracy_by_layer: Dict[int, float],
        languages: List[str],
        save_path: Path | str,
    ) -> None:
        """Plot logit-lens decoding accuracy per layer for multiple languages."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        layers = sorted(accuracy_by_layer.keys())

        # Support single accuracy dict or nested per-language accuracy
        first_val = next(iter(accuracy_by_layer.values()))
        if isinstance(first_val, dict):
            for lang in languages:
                color = get_language_color(lang)
                vals = [accuracy_by_layer[l].get(lang, 0.0) for l in layers]
                ax.plot(layers, vals, marker="o", label=lang.upper(), color=color, linewidth=1.8)
        else:
            # Single accuracy curve
            vals = [accuracy_by_layer[l] for l in layers]
            ax.plot(layers, vals, marker="o", color="blue", linewidth=2.0, label="Accuracy")

        ax.set_xlabel("Transformer Layer ID")
        ax.set_ylabel("Logit-Lens Accuracy")
        ax.set_title("Layer-wise Logit-Lens Task Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved logit-lens accuracy plot to %s", save_path)

    def plot_weight_matrix_svd(
        self,
        weight_matrix: Tensor,
        layer_id: int,
        save_path: Path | str,
    ) -> None:
        """Plot the SVD spectrum of a model weight matrix to inspect rank/energy."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        W = weight_matrix.detach().cpu().float()
        # Compute singular values
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        s_vals = S.numpy()

        energy = (s_vals ** 2) / np.sum(s_vals ** 2)
        cum_energy = np.cumsum(energy)

        ax.plot(range(1, len(s_vals) + 1), cum_energy, color="teal", label="Cumulative energy", linewidth=2.0)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Cumulative Spectral Energy")
        ax.set_title(f"Weight SVD Spectrum: Layer {layer_id}")
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved weight SVD spectrum to %s", save_path)

    def plot_token_logit_evolution(
        self,
        logits_by_layer: Dict[int, Tensor],
        target_tokens: List[int],
        tokenizer,
        save_path: Path | str,
    ) -> None:
        """Plot how logits of target tokens evolve across layers."""
        fig, ax = plt.subplots(figsize=self.config.figsize_default)

        layers = sorted(logits_by_layer.keys())

        for tok_id in target_tokens:
            token_str = tokenizer.decode([tok_id])
            vals = []
            for l in layers:
                # Shape of logits at layer: (vocab_size)
                logits = logits_by_layer[l]
                vals.append(logits[tok_id].item())
            ax.plot(layers, vals, marker="s", label=f"'{token_str}' ({tok_id})", linewidth=1.8)

        ax.set_xlabel("Transformer Layer ID")
        ax.set_ylabel("Logit Value")
        ax.set_title("Layer-wise Logit Evolution for Key Tokens")
        ax.legend()
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved token logit evolution plot to %s", save_path)

    def plot_hidden_state_cosine_sim_matrix(
        self,
        states: Dict[str, Tensor],
        layer_id: int,
        save_path: Path | str,
    ) -> None:
        """Plot the pairwise cosine similarity matrix between language states at a layer."""
        fig, ax = plt.subplots(figsize=(7, 6))

        languages = list(states.keys())
        matrix = np.zeros((len(languages), len(languages)))

        for i, lang_a in enumerate(languages):
            h_a = states[lang_a].float().mean(dim=0)  # (hidden_dim)
            for j, lang_b in enumerate(languages):
                h_b = states[lang_b].float().mean(dim=0)
                sim = torch.nn.functional.cosine_similarity(h_a, h_b, dim=0).item()
                matrix[i, j] = sim

        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            xticklabels=[l.upper() for l in languages],
            yticklabels=[l.upper() for l in languages],
            ax=ax,
            vmin=-1,
            vmax=1,
        )

        ax.set_title(f"Pairwise Language Cosine Similarity Matrix (Layer {layer_id})")

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved hidden state cosine similarity matrix to %s", save_path)

    def plot_pca_3d(
        self,
        states: Dict[str, Tensor],
        layer_id: int,
        save_path: Path | str,
    ) -> None:
        """Plot 3D PCA scatter using plotly if available, else static matplotlib 3D scatter."""
        save_path = Path(save_path)
        out_dir = save_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            import plotly.graph_objects as go
            import pandas as pd
            from sklearn.decomposition import PCA

            # Gather all states into single dataset
            all_states = []
            labels = []
            for lang, s in states.items():
                s_np = s.numpy() if isinstance(s, Tensor) else np.array(s)
                all_states.append(s_np)
                labels.extend([lang.upper()] * len(s_np))

            X = np.concatenate(all_states, axis=0)
            pca = PCA(n_components=3)
            X_3d = pca.fit_transform(X)

            df = pd.DataFrame(X_3d, columns=["PC1", "PC2", "PC3"])
            df["Language"] = labels

            fig = go.Figure()
            for lang in df["Language"].unique():
                df_sub = df[df["Language"] == lang]
                color = get_language_color(lang.lower())
                fig.add_trace(go.Scatter3d(
                    x=df_sub["PC1"],
                    y=df_sub["PC2"],
                    z=df_sub["PC3"],
                    mode="markers",
                    name=lang,
                    marker=dict(size=4, color=color, opacity=0.8)
                ))

            fig.update_layout(
                title=f"3D PCA of Language Representations (Layer {layer_id})",
                scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"),
                margin=dict(r=0, l=0, b=0, t=40)
            )

            html_path = out_dir / f"{save_path.stem}.html"
            fig.write_html(str(html_path))
            logger.info("Saved interactive 3D PCA plot to %s", html_path)
        except ImportError:
            logger.warning("plotly/sklearn not installed; plotting static 3D PCA using matplotlib.")

        # Always save static matplotlib 3D scatter as fallback/requirement
        fig = plt.figure(figsize=self.config.figsize_default)
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection="3d")

        from sklearn.decomposition import PCA
        for lang, s in states.items():
            s_np = s.numpy() if isinstance(s, Tensor) else np.array(s)
            pca = PCA(n_components=3)
            s_3d = pca.fit_transform(s_np)
            color = get_language_color(lang)
            ax.scatter(s_3d[:, 0], s_3d[:, 1], s_3d[:, 2], label=lang.upper(), color=color, alpha=0.7)

        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        ax.set_title(f"3D PCA of Representations (Layer {layer_id})")
        ax.legend()

        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved static 3D PCA plot to %s", save_path)

    # ── Logit-distill Visualizations Alignment ───────────────────────────────

    def plot_attention_entropy(
        self,
        attn_weights_list: List[List[np.ndarray]],
        model_labels: List[str],
        save_path: Path | str,
    ) -> None:
        """Box plots of attention entropy per layer across all heads for capacity bottleneck diagnostics."""
        n_layers = len(attn_weights_list[0])
        fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.9), 5))

        cmap = plt.cm.tab10.colors
        width = 0.8 / len(attn_weights_list)

        for midx, (layers, label) in enumerate(zip(attn_weights_list, model_labels)):
            entropies_per_layer = []
            for arr in layers:
                arr_f = to_numpy(arr).astype(np.float64)
                p = arr_f + 1e-12
                ent = -np.sum(p * np.log(p), axis=-1)
                entropies_per_layer.append(ent.ravel())

            positions = [li + midx * width - 0.4 + width / 2 for li in range(n_layers)]
            ax.boxplot(
                entropies_per_layer,
                positions=positions,
                widths=width * 0.85,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor=cmap[midx % 10], alpha=0.6),
                medianprops=dict(color="black", lw=1.5),
            )
            ax.plot([], [], color=cmap[midx % 10], label=label, lw=6, alpha=0.6)

        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f"L{i}" for i in range(n_layers)], fontsize=8)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Attention Entropy (nats)")
        ax.set_title("Per-Layer Attention Entropy across Heads")
        ax.legend(fontsize=9)
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved attention entropy box plots to %s", save_path)

    def plot_attention_patterns_grid(
        self,
        attn_weights: np.ndarray | Tensor,
        layer_id: int,
        save_path: Path | str,
        example_idx: int = 0,
    ) -> None:
        """Grid plot showing attention pattern heads of a chosen layer side-by-side."""
        arr = to_numpy(attn_weights)
        n_heads = arr.shape[1]
        ncols = min(8, n_heads)
        nrows = (n_heads + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
        axes_flat = np.array(axes).ravel()

        for h in range(n_heads):
            mat = arr[example_idx, h]
            axes_flat[h].imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=mat.max())
            axes_flat[h].set_title(f"Head {h}", fontsize=8)
            axes_flat[h].set_xticks([])
            axes_flat[h].set_yticks([])

        for h in range(n_heads, len(axes_flat)):
            axes_flat[h].set_visible(False)

        fig.suptitle(f"All Attention Heads — Layer {layer_id}", fontsize=11, y=1.01)
        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved attention patterns grid to %s", save_path)

    def plot_layer_norms_comparison(
        self,
        states_before: List[np.ndarray],
        states_after: List[np.ndarray],
        labels: List[str],
        save_path: Path | str,
    ) -> None:
        """Plot showing absolute norms and relative depth norms of hidden states before and after interventions."""
        t_norms = [np.linalg.norm(hs, axis=-1).mean() for hs in states_before]
        s_norms = [np.linalg.norm(hs, axis=-1).mean() for hs in states_after]

        t_layers = np.arange(len(t_norms))
        s_layers = np.arange(len(s_norms))

        t_x = t_layers / max(len(t_norms) - 1, 1)
        s_x = s_layers / max(len(s_norms) - 1, 1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        ax.plot(t_layers, t_norms, "o-", color="#8c564b", lw=2, ms=5, label=labels[0])
        ax.plot(s_layers, s_norms, "s--", color="#e377c2", lw=2, ms=5, label=labels[1])
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Mean L2 Norm")
        ax.set_title("Absolute Layer Norms")
        ax.legend()
        ax.grid(True)

        ax = axes[1]
        ax.plot(t_x, t_norms, "o-", color="#8c564b", lw=2, ms=5, label=labels[0])
        ax.plot(s_x, s_norms, "s--", color="#e377c2", lw=2, ms=5, label=labels[1])
        ax.set_xlabel("Relative Depth (0=embedding, 1=final)")
        ax.set_ylabel("Mean L2 Norm")
        ax.set_title("Layer Norms at Matched Relative Depth")
        ax.legend()
        ax.grid(True)

        fig.suptitle("Activation L2 Norm Evolution Across Depth", fontsize=11, y=1.01)
        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved layer norms comparison to %s", save_path)

    def plot_activation_statistics_ribbon(
        self,
        hidden_states_list: List[List[np.ndarray]],
        labels: List[str],
        stat: str,
        save_path: Path | str,
    ) -> None:
        """Plot a ribbon chart of mean/std of activations per layer for diagnostics."""
        fig, ax = plt.subplots(figsize=(10, 4.5))
        cmap = plt.cm.tab10.colors

        for midx, (layers, label) in enumerate(zip(hidden_states_list, labels)):
            layer_means, layer_stds = [], []
            for hs in layers:
                if stat == "norm":
                    vals = np.linalg.norm(hs.astype(np.float64), axis=-1)
                elif stat == "mean":
                    vals = hs.astype(np.float64).mean(axis=-1)
                else:
                    vals = hs.astype(np.float64).std(axis=-1)
                layer_means.append(vals.mean())
                layer_stds.append(vals.std())

            xs = np.arange(len(layers))
            means = np.array(layer_means)
            stds = np.array(layer_stds)
            color = cmap[midx % len(cmap)]
            ax.plot(xs, means, "o-", color=color, lw=2, ms=4, label=label)
            ax.fill_between(xs, means - stds, means + stds, alpha=0.18, color=color)

        stat_labels = {"norm": "Mean L2 Norm", "mean": "Mean Activation", "std": "Std of Activations"}
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(stat_labels.get(stat, stat))
        ax.set_title(f"Activation Statistics Ribbon Chart ({stat_labels.get(stat, stat)})")
        ax.legend(fontsize=9)
        ax.grid(True)

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved activation statistics ribbon chart to %s", save_path)

    def plot_tokenizer_alignment(
        self,
        tokenizer_src,
        tokenizer_tgt,
        texts: List[str],
        save_path: Path | str,
        n_examples: int = 4,
    ) -> None:
        """Visual comparison diff of tokenizations between languages / model architectures."""
        n = min(n_examples, len(texts))
        fig, axes = plt.subplots(n, 1, figsize=(14, max(3, n * 2.0)), squeeze=False)

        for i, text in enumerate(texts[:n]):
            ax = axes[i, 0]
            ax.axis("off")

            src_tokens = tokenizer_src.tokenize(text)
            tgt_tokens = tokenizer_tgt.tokenize(text)

            def _draw_tokens(token_list, y_pos, label, color):
                x = 0.0
                ax.text(-0.01, y_pos, label, ha="right", va="center", fontsize=8, fontweight="bold", transform=ax.transAxes)
                for tok in token_list:
                    display = tok.replace("▁", "_").replace("Ġ", "_").replace(" ", "_")
                    width = max(0.03, len(display) * 0.013)
                    rect = plt.Rectangle((x, y_pos - 0.12), width, 0.22, transform=ax.transAxes, color=color, alpha=0.35, clip_on=False)
                    ax.add_patch(rect)
                    ax.text(x + width / 2, y_pos, display, ha="center", va="center", fontsize=7, transform=ax.transAxes, clip_on=False)
                    x += width + 0.004
                    if x > 0.95:
                        break

            _draw_tokens(src_tokens, 0.7, "English/Src", "#1f77b4")
            _draw_tokens(tgt_tokens, 0.3, "Target", "#ff7f0e")

            n_match = sum(1 for st, tt in zip(src_tokens, tgt_tokens) if st == tt)
            pct = 100 * n_match / max(len(src_tokens), 1)
            ax.set_title(f'"{text[:60]}…"  |  Src: {len(src_tokens)} tokens, Tgt: {len(tgt_tokens)} tokens, {pct:.0f}% match', fontsize=8)

        fig.suptitle("Cross-Lingual/Cross-Model Tokenizer Alignment Diff", fontsize=11, y=1.01)
        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved tokenizer alignment diff plot to %s", save_path)

    def plot_vocab_overlap(
        self,
        tokenizer_src,
        tokenizer_tgt,
        save_path: Path | str,
    ) -> None:
        """Venn-diagram-style bar chart of vocabulary sizes and overlap between tokenizers."""
        src_vocab = set(range(tokenizer_src.vocab_size))
        tgt_vocab = set(range(tokenizer_tgt.vocab_size))
        overlap = src_vocab & tgt_vocab
        src_only = src_vocab - tgt_vocab
        tgt_only = tgt_vocab - src_vocab

        labels = ["Source Only", "Shared Overlap", "Target Only"]
        values = [len(src_only), len(overlap), len(tgt_only)]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(range(3), values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Vocabulary Size (Tokens)")
        ax.set_title("Tokenizer Vocabulary Overlap Analysis")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500, f"{val:,}", ha="center", va="bottom", fontsize=8)

        pct = 100 * len(overlap) / max(len(src_vocab | tgt_vocab), 1)
        ax.text(0.98, 0.96, f"Overlap: {pct:.1f}%  |  Src: {len(src_vocab):,}  Tgt: {len(tgt_vocab):,}",
                ha="right", va="top", transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        save_path = Path(save_path)
        save_figure(fig, save_path, self.config)
        plt.close(fig)
        logger.info("Saved vocabulary overlap analysis to %s", save_path)
