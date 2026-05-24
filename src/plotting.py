"""Gráficas del estudio Monte Carlo (Test 1, estadístico T_n)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap


# ---------------------------------------------------------------------------
# Utilidad
# ---------------------------------------------------------------------------
def _safe_filename(s: str) -> str:
    return (
        s.replace("(", "_").replace(")", "")
        .replace(",", "_").replace("=", "")
        .replace(" ", "").replace(".", "p")
    )


def _sigma_band_heatmap(
    ax,
    pivot: pd.DataFrame,
    R: int,
    alpha: float = 0.05,
    title: str = "",
):
    """Heatmap de Error Tipo I con colormap discreto por bandas σ.

    Replica el estilo del Proyecto 1: verde en la diana (|err - α| < 1σ),
    amarillo pálido en ±1–2σ, rojo en los extremos.
    """
    mu = alpha * 100
    sigma = np.sqrt(alpha * (1 - alpha) / R) * 100

    limites = [
        0,
        mu - 3 * sigma,
        mu - 2 * sigma,
        mu - 1 * sigma,
        mu + 1 * sigma,
        mu + 2 * sigma,
        mu + 3 * sigma,
        100,
    ]
    colores = ["darkred", "tomato", "khaki", "seagreen", "khaki", "tomato", "darkred"]
    cmap_d = ListedColormap(colores)
    norm = BoundaryNorm(limites, cmap_d.N)

    tick_pos = [(limites[i] + limites[i + 1]) / 2 for i in range(len(limites) - 1)]
    tick_lbl = [
        r"$>3\sigma$ (muy bajo)",
        r"$2\sigma$–$3\sigma$ (bajo)",
        r"$1\sigma$–$2\sigma$",
        r"$<1\sigma$ (diana)",
        r"$1\sigma$–$2\sigma$",
        r"$2\sigma$–$3\sigma$ (alto)",
        r"$>3\sigma$ (muy alto)",
    ]

    sns.set_style("white")
    hm = sns.heatmap(
        pivot * 100,
        annot=True, fmt=".1f",
        cmap=cmap_d, norm=norm,
        linewidths=0.5, linecolor="white",
        cbar_kws={"ticks": tick_pos, "label": f"Evaluación respecto al {int(alpha*100)}% teórico"},
        ax=ax,
    )
    hm.collections[0].colorbar.set_ticklabels(tick_lbl)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Estimador θ̃", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)


# ---------------------------------------------------------------------------
# Error Tipo I — líneas
# ---------------------------------------------------------------------------
def plot_type_i_error(
    summary: pd.DataFrame,
    alpha: float,
    outdir: Path,
) -> Path:
    """Tasa de Error Tipo I vs n, una curva por estimador y distribución H0."""
    h0 = summary[summary["under_h0"]].copy()
    if h0.empty:
        return None
    dists = sorted(h0["dist"].unique())
    estimators = sorted(h0["estimator"].unique())

    fig, axes = plt.subplots(
        1, len(dists), figsize=(5.5 * len(dists), 4.2), sharey=True, squeeze=False
    )
    for ax, d in zip(axes[0], dists):
        sub = h0[h0["dist"] == d]
        for est in estimators:
            s2 = sub[sub["estimator"] == est].sort_values("n")
            if s2.empty:
                continue
            ax.errorbar(
                s2["n"], s2["reject_rate"], yerr=2 * s2["se_rate"],
                marker="o", capsize=3, linewidth=2, label=est,
            )
        ax.axhline(alpha, color="black", linestyle="--", linewidth=1,
                   label=f"nivel α={alpha}")
        ax.set_xscale("log")
        ax.set_xlabel("n (escala log)")
        ax.set_title(d)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].set_ylabel("Tasa de rechazo (Error Tipo I)")
    axes[0, -1].legend(loc="best", fontsize=9)
    fig.suptitle("Error Tipo I bajo H_0 — Test T_n", y=1.02)
    fig.tight_layout()
    out = outdir / "tn_type_i_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Error Tipo I — heatmap sigma-band
# ---------------------------------------------------------------------------
def plot_type_i_error_heatmap(
    summary: pd.DataFrame,
    alpha: float,
    outdir: Path,
    R: int = 300,
) -> Path | None:
    """Heatmap de Error Tipo I con bandas σ (estilo Proyecto 1)."""
    h0 = summary[summary["under_h0"]].copy()
    if h0.empty:
        return None

    pivot = h0.pivot_table(index=["dist", "n"], columns="estimator",
                           values="reject_rate")
    pivot.index = [f"{d} | n={n}" for d, n in pivot.index]

    fig, ax = plt.subplots(
        figsize=(max(6, 2.2 * len(pivot.columns)), max(4, 0.45 * len(pivot.index) + 1.5))
    )
    _sigma_band_heatmap(
        ax, pivot, R=R, alpha=alpha,
        title=f"Error Tipo I (%) — T_n  (esperado ≈ {int(alpha*100)}%)",
    )
    fig.tight_layout()
    out = outdir / "tn_type_i_error_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Potencia — líneas
# ---------------------------------------------------------------------------
def plot_power_curves(
    summary: pd.DataFrame,
    outdir: Path,
) -> Path | None:
    """Curvas de potencia vs n por estimador, una sub-gráfica por distribución Ha."""
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty:
        return None
    dists = sorted(ha["dist"].unique())
    estimators = sorted(ha["estimator"].unique())

    n_cols = min(3, len(dists))
    n_rows = int(np.ceil(len(dists) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    for ax, d in zip(axes_flat, dists):
        sub = ha[ha["dist"] == d]
        for est in estimators:
            s2 = sub[sub["estimator"] == est].sort_values("n")
            if s2.empty:
                continue
            ax.errorbar(
                s2["n"], s2["reject_rate"], yerr=2 * s2["se_rate"],
                marker="o", capsize=3, linewidth=2, label=est,
            )
        ax.set_xscale("log")
        ax.set_xlabel("n (escala log)")
        ax.set_title(d)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
    for ax in axes_flat[len(dists):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Potencia empírica")
    axes_flat[0].legend(loc="best", fontsize=9)
    fig.suptitle("Potencia bajo H_a — Test T_n", y=1.02)
    fig.tight_layout()
    out = outdir / "tn_power_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Potencia — heatmap RdYlGn
# ---------------------------------------------------------------------------
def plot_power_heatmap(
    summary: pd.DataFrame,
    outdir: Path,
) -> Path | None:
    """Heatmap de potencia con cmap RdYlGn (estilo Proyecto 1)."""
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty:
        return None

    pivot = ha.pivot_table(index=["dist", "n"], columns="estimator",
                           values="reject_rate") * 100
    pivot.index = [f"{d} | n={n}" for d, n in pivot.index]

    fig, ax = plt.subplots(
        figsize=(max(6, 2.2 * len(pivot.columns)), max(4, 0.45 * len(pivot.index) + 1.5))
    )
    sns.set_style("white")
    sns.heatmap(
        pivot,
        annot=True, fmt=".1f",
        cmap="RdYlGn",
        vmin=0, vmax=100,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Potencia empírica (%)"},
        ax=ax,
    )
    ax.set_title("Potencia empírica (%) — T_n  (mayor es mejor)", pad=12)
    ax.set_xlabel("Estimador θ̃", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    out = outdir / "tn_power_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
def plot_runtime(summary: pd.DataFrame, outdir: Path) -> Path:
    """Tiempo medio por test vs n, por estimador."""
    df = summary.copy()
    g = df.groupby(["n", "estimator"], as_index=False).agg(
        mean_time_s=("mean_time_s", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for est in sorted(g["estimator"].unique()):
        s2 = g[g["estimator"] == est].sort_values("n")
        ax.plot(s2["n"], s2["mean_time_s"], marker="o", linewidth=2, label=est)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log)")
    ax.set_ylabel("Tiempo medio por test [s] (log)")
    ax.set_title("Costo computacional del test T_n (B remuestras)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = outdir / "tn_runtime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Potencia vs costo
# ---------------------------------------------------------------------------
def plot_power_vs_cost(summary: pd.DataFrame, outdir: Path) -> Path | None:
    """Scatter potencia vs costo computacional, colores por estimador."""
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    colors = {"argmin": "C0", "median": "C1", "trimmed": "C2"}
    dists = sorted(ha["dist"].unique())
    estimators = sorted(ha["estimator"].unique())
    for i, d in enumerate(dists):
        mk = markers[i % len(markers)]
        for est in estimators:
            sub = ha[(ha["dist"] == d) & (ha["estimator"] == est)]
            ax.scatter(
                sub["mean_time_s"], sub["reject_rate"],
                marker=mk, color=colors.get(est, "k"),
                s=50, alpha=0.85,
                label=f"{d} | {est}" if i == 0 else None,
            )
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=e)
               for e, c in colors.items() if e in estimators]
    ax.legend(handles=handles, title="Estimador θ̃", loc="best")
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo medio por test [s] (log)")
    ax.set_ylabel("Potencia empírica")
    ax.set_title("Costo computacional vs Potencia (T_n bajo H_a)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out = outdir / "tn_power_vs_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Distribución del p-valor bajo H0
# ---------------------------------------------------------------------------
def plot_pvalue_distribution_h0(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Histograma del p-valor bajo H0 (debería ser ~uniforme en (0,1))."""
    h0 = df[df["under_h0"]].copy()
    if h0.empty:
        return None
    estimators = sorted(h0["estimator"].unique())
    dists = sorted(h0["dist"].unique())
    fig, axes = plt.subplots(
        len(dists), len(estimators),
        figsize=(3.5 * len(estimators), 2.8 * len(dists)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, d in enumerate(dists):
        for j, est in enumerate(estimators):
            ax = axes[i, j]
            sub = h0[(h0["dist"] == d) & (h0["estimator"] == est)]
            ax.hist(sub["p_value"], bins=20, range=(0, 1),
                    color="C0", edgecolor="white", density=True)
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{d} | {est}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("p-valor")
    for ax in axes[:, 0]:
        ax.set_ylabel("densidad")
    fig.suptitle("Distribución del p-valor bajo H_0 (T_n)", y=1.02)
    fig.tight_layout()
    out = outdir / "tn_pvalue_h0.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
