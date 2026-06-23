"""
Publication-quality visualization utilities for the Causal Climate Policy project.

Provides consistent styling, color palettes, and figure generation functions
for all plots in the paper manuscript.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# ── Publication Style Configuration ──────────────────────────────────────────

# Color palette — accessible, journal-friendly
COLORS = {
    "primary": "#2563EB",       # Blue
    "secondary": "#7C3AED",     # Purple
    "success": "#059669",       # Green
    "warning": "#D97706",       # Amber
    "danger": "#DC2626",        # Red
    "neutral": "#6B7280",       # Gray
    "carbon_tax": "#2563EB",
    "renewable_mandate": "#059669",
    "efficiency_standard": "#D97706",
    "subsidy": "#7C3AED",
}

SECTOR_COLORS = {
    "Power": "#2563EB",
    "Transport": "#DC2626",
    "Industry": "#D97706",
    "Agriculture": "#059669",
    "Buildings": "#7C3AED",
}

POLICY_MARKERS = {
    "Carbon Tax": "o",
    "Renewable Mandate": "s",
    "Efficiency Standard": "D",
    "Subsidy": "^",
}


def set_publication_style():
    """
    Set matplotlib/seaborn defaults for publication-quality figures.
    Call once at the top of each visualization notebook.
    """
    plt.rcParams.update({
        # Figure
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,

        # Font
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        # Axes
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
    })
    sns.set_palette([
        COLORS["primary"],
        COLORS["success"],
        COLORS["warning"],
        COLORS["danger"],
        COLORS["secondary"],
        COLORS["neutral"],
    ])


def save_figure(fig, name: str, formats=("png", "pdf"), subdir: str = None):
    """
    Save a matplotlib figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Base filename (without extension).
    formats : tuple
        File formats to save (default: PNG and PDF).
    subdir : str, optional
        Subdirectory within outputs/figures/.
    """
    base_dir = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = base_dir / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches="tight", dpi=300)
        print(f"  Saved: {filepath}")


def plot_treatment_effects(
    effects_df: pd.DataFrame,
    title: str = "Average Treatment Effects by Policy Type",
    figsize: tuple = (8, 5),
):
    """
    Create a forest plot of treatment effects with confidence intervals.

    Parameters
    ----------
    effects_df : pd.DataFrame
        Must contain columns: 'policy_type', 'estimate', 'ci_lower', 'ci_upper'
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    y_positions = range(len(effects_df))
    colors = [COLORS.get(pt.lower().replace(" ", "_"), COLORS["neutral"])
              for pt in effects_df["policy_type"]]

    # Plot confidence intervals
    for i, (_, row) in enumerate(effects_df.iterrows()):
        ax.hlines(i, row["ci_lower"], row["ci_upper"],
                  color=colors[i], linewidth=2, alpha=0.7)
        ax.plot(row["estimate"], i, "o", color=colors[i],
                markersize=8, zorder=5)

    # Zero line
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(effects_df["policy_type"])
    ax.set_xlabel("Estimated Effect on Log Emissions")
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_cate_heterogeneity(
    cate_df: pd.DataFrame,
    x_var: str,
    x_label: str,
    title: str = "Heterogeneous Treatment Effects",
    figsize: tuple = (8, 5),
):
    """
    Scatter plot of CATEs against a moderating variable with LOWESS trend.

    Parameters
    ----------
    cate_df : pd.DataFrame
        Must contain columns: x_var, 'cate', 'cate_lower', 'cate_upper'
    x_var : str
        Column name for the x-axis moderator.
    x_label : str
        Human-readable label for x-axis.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(cate_df[x_var], cate_df["cate"],
               alpha=0.4, s=20, color=COLORS["primary"])

    # Confidence band
    ax.fill_between(
        cate_df[x_var].sort_values(),
        cate_df.sort_values(x_var)["cate_lower"],
        cate_df.sort_values(x_var)["cate_upper"],
        alpha=0.15, color=COLORS["primary"],
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Conditional Average Treatment Effect")
    ax.set_title(title)

    plt.tight_layout()
    return fig
