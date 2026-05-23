"""Gráficas del estudio Monte Carlo (Test 1, estadístico T_n)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_filename(s: str) -> str:
    return (
        s.replace("(", "_").replace(")", "")
        .replace(",", "_").replace("=", "")
        .replace(" ", "").replace(".", "p")
    )


def plot_type_i_error(
    summary: pd.DataFrame,
    alpha: float,
    outdir: Path,
) -> Path:
    """Tasa de Error Tipo I vs n, una curva por estimador y por distribución H0.

    Incluye banda de fluctuación de 2 SE alrededor de la tasa estimada.
    """
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
                marker="o", capsize=3, label=est,
            )
        ax.axhline(alpha, color="black", linestyle="--", linewidth=1, label=f"nivel α={alpha}")
        ax.set_xscale("log")
        ax.set_xlabel("n (escala log)")
        ax.set_title(d)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Tasa de rechazo (Error Tipo I)")
    axes[0, -1].legend(loc="best", fontsize=9)
    fig.suptitle("Error Tipo I bajo H0 — Test T_n", y=1.02)
    fig.tight_layout()
    out = outdir / "tn_type_i_error.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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
                marker="o", capsize=3, label=est,
            )
        ax.set_xscale("log")
        ax.set_xlabel("n (escala log)")
        ax.set_title(d)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
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


def plot_runtime(summary: pd.DataFrame, outdir: Path) -> Path:
    """Tiempo medio por test vs n, por estimador (promedio sobre distribuciones)."""
    df = summary.copy()
    g = df.groupby(["n", "estimator"], as_index=False).agg(
        mean_time_s=("mean_time_s", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for est in sorted(g["estimator"].unique()):
        s2 = g[g["estimator"] == est].sort_values("n")
        ax.plot(s2["n"], s2["mean_time_s"], marker="o", label=est)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log)")
    ax.set_ylabel("Tiempo medio por test [s] (log)")
    ax.set_title("Costo computacional del test T_n (B remuestras)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = outdir / "tn_runtime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_power_vs_cost(summary: pd.DataFrame, outdir: Path) -> Path | None:
    """Scatter potencia vs costo computacional, colores por estimador, marca por dist."""
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
                s=40, alpha=0.85,
                label=f"{d} | {est}" if i == 0 else None,
            )
    # Una leyenda más limpia: por estimador
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=c, label=e)
               for e, c in colors.items() if e in estimators]
    ax.legend(handles=handles, title="Estimador θ̃", loc="best")
    ax.set_xscale("log")
    ax.set_xlabel("Tiempo medio por test [s] (log)")
    ax.set_ylabel("Potencia empírica")
    ax.set_title("Costo computacional vs Potencia (T_n bajo H_a)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = outdir / "tn_power_vs_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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
    for ax in axes[-1]:
        ax.set_xlabel("p-valor")
    for ax in axes[:, 0]:
        ax.set_ylabel("densidad")
    fig.suptitle("Distribución del p-valor bajo H0 (T_n)", y=1.02)
    fig.tight_layout()
    out = outdir / "tn_pvalue_h0.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
