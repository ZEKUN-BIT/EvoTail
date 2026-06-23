"""Matplotlib figures for Phase 1; imported only on a plotting-capable node."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot_support_gap(
    assay_summary: pd.DataFrame, stratum_summary: pd.DataFrame, output_path: str | Path
) -> None:
    """Save a scatter-plus-bootstrap-bar Figure 1 panel."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on cluster environment
        raise RuntimeError("Install matplotlib to generate figures") from error

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].scatter(
        assay_summary["evolutionary_support_score"], assay_summary["spearman"], alpha=0.7
    )
    axes[0].set_xlabel("Evolutionary support score")
    axes[0].set_ylabel("Per-assay zero-shot Spearman")
    axes[0].set_title("Observed support-performance association")

    errors = [
        stratum_summary["mean_spearman"] - stratum_summary["ci_low"],
        stratum_summary["ci_high"] - stratum_summary["mean_spearman"],
    ]
    axes[1].bar(stratum_summary["support_stratum"].astype(str), stratum_summary["mean_spearman"], yerr=errors, capsize=3)
    axes[1].set_xlabel("Support stratum (low → high)")
    axes[1].set_ylabel("Mean per-assay Spearman")
    axes[1].set_title("Bootstrap 95% confidence intervals")
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

