"""Figuras de calidad de estimadores del centro de simetría (apéndice)."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

COLORS = {
    "argmin":  "#2196F3",   # azul
    "median":  "#4CAF50",   # verde
    "trimmed": "#9C27B0",   # morado
}
LABELS = {
    "argmin":  r"$\hat\theta_{\min}$ (argmin $T_n$)",
    "median":  r"$\hat\theta_{\rm med}$ (mediana)",
    "trimmed": r"$\hat\theta_\alpha$ (media afeitada)",
}
MARKERS = {"argmin": "o", "median": "^", "trimmed": "D"}


def plot_rmse_h0(summary: pd.DataFrame, outdir: Path) -> Path:
    """RMSE vs n por estimador bajo H0, panel por distribución nula."""
    h0 = summary[summary["under_h0"]].copy()
    dists = sorted(h0["dist"].unique())
    ests  = ["argmin", "median", "trimmed"]

    fig, axes = plt.subplots(1, len(dists), figsize=(5.5 * len(dists), 4.2),
                             sharey=False, squeeze=False)
    for ax, d in zip(axes[0], dists):
        sub = h0[h0["dist"] == d]
        for est in ests:
            s2 = sub[sub["estimator"] == est].sort_values("n")
            ax.plot(s2["n"], s2["rmse"], marker=MARKERS[est],
                    color=COLORS[est], linewidth=2, label=LABELS[est])
        ax.set_xscale("log")
        ax.set_xlabel("n (escala log)", fontsize=10)
        ax.set_ylabel("RMSE", fontsize=10)
        ax.set_title(d, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks([20, 40, 80, 160])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle(r"RMSE de $\hat\theta$ bajo $H_0$ (centro verdadero $\theta=2$)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = outdir / "apendice_rmse_h0.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_bias_h0(summary: pd.DataFrame, outdir: Path) -> Path:
    """Sesgo vs n por estimador bajo H0, panel por distribución nula."""
    h0 = summary[summary["under_h0"]].copy()
    dists = sorted(h0["dist"].unique())
    ests  = ["argmin", "median", "trimmed"]

    fig, axes = plt.subplots(1, len(dists), figsize=(5.5 * len(dists), 4.2),
                             sharey=True, squeeze=False)
    for ax, d in zip(axes[0], dists):
        sub = h0[h0["dist"] == d]
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        for est in ests:
            s2 = sub[sub["estimator"] == est].sort_values("n")
            ax.plot(s2["n"], s2["bias"], marker=MARKERS[est],
                    color=COLORS[est], linewidth=2, label=LABELS[est])
        ax.set_xscale("log")
        ax.set_xlabel("n (escala log)", fontsize=10)
        ax.set_ylabel("Sesgo", fontsize=10)
        ax.set_title(d, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks([20, 40, 80, 160])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0, -1].legend(loc="best", fontsize=9)
    fig.suptitle(r"Sesgo de $\hat\theta$ bajo $H_0$ (centro verdadero $\theta=2$)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = outdir / "apendice_bias_h0.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mean_ha(summary: pd.DataFrame, outdir: Path) -> Path:
    """Media ± std de theta_hat bajo Ha por estimador y distribución."""
    ha = summary[~summary["under_h0"]].copy()
    dists = sorted(ha["dist"].unique())
    ests  = ["argmin", "median", "trimmed"]

    n_cols = len(dists)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.2),
                             squeeze=False)
    for ax, d in zip(axes[0], dists):
        sub = ha[ha["dist"] == d].sort_values("n")
        for est in ests:
            s2 = sub[sub["estimator"] == est].sort_values("n")
            ax.errorbar(
                s2["n"], s2["mean_hat"], yerr=s2["std_hat"],
                marker=MARKERS[est], color=COLORS[est],
                linewidth=2, capsize=3, label=LABELS[est],
            )
        ax.set_xscale("log")
        ax.set_xlabel("n (escala log)", fontsize=10)
        ax.set_ylabel(r"$\bar{\hat\theta}$ (media de las estimaciones)", fontsize=10)
        ax.set_title(d, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks([20, 40, 80, 160])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        r"Media $\pm$ std de $\hat\theta$ bajo $H_a$ (distribuciones asimétricas)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = outdir / "apendice_theta_ha.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rmse_heatmap(summary: pd.DataFrame, outdir: Path) -> Path:
    """Heatmap de RMSE bajo H0 con colores relativos."""
    h0 = summary[summary["under_h0"]].copy()
    pivot = h0.pivot_table(index=["dist", "n"], columns="estimator",
                           values="rmse")
    pivot = pivot[["argmin", "median", "trimmed"]]
    pivot.index = [f"{d} | n={n}" for d, n in pivot.index]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(pivot) + 1.5)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f",
        cmap="YlOrRd_r",          # amarillo (bajo RMSE) a rojo (alto)
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "RMSE"},
        ax=ax,
    )
    ax.set_title("RMSE de los estimadores bajo $H_0$ — menor es mejor",
                 pad=12, fontsize=12)
    ax.set_xlabel("Estimador", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    out = outdir / "apendice_rmse_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_combined_appendix(summary: pd.DataFrame, outdir: Path) -> Path:
    """Figura combinada 2×2 para el apéndice: RMSE H0 + media Ha + heatmap."""
    h0 = summary[summary["under_h0"]].copy()
    ha = summary[~summary["under_h0"]].copy()
    ests = ["argmin", "median", "trimmed"]
    ns   = sorted(h0["n"].unique())

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Panel A: RMSE bajo H0 (Uniforme y Cauchy) -------------------------
    ax_a = [fig.add_subplot(gs[0, j]) for j in range(2)]
    for ax, d in zip(ax_a, sorted(h0["dist"].unique())):
        sub = h0[h0["dist"] == d]
        for est in ests:
            s2 = sub[sub["estimator"] == est].sort_values("n")
            ax.plot(s2["n"], s2["rmse"], marker=MARKERS[est],
                    color=COLORS[est], linewidth=2, label=LABELS[est])
        ax.set_xscale("log")
        ax.set_xticks(ns); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("n", fontsize=9); ax.set_ylabel("RMSE", fontsize=9)
        ax.set_title(f"(A) RMSE bajo $H_0$: {d}", fontsize=10)
        ax.grid(True, ls="--", alpha=0.4)
    ax_a[0].legend(loc="upper right", fontsize=8)

    # ---- Panel B: Media ± 1 std de θ̂ bajo Ha --------------------------------
    ax_b = [fig.add_subplot(gs[1, j]) for j in range(2)]
    ha_dists = sorted(ha["dist"].unique())
    # Dos distribuciones en los paneles B; si hay 3, elige Gamma y Pareto
    show_dists = [d for d in ["Gamma(k=2.0,s=1.0)", "Pareto(a=3.0,s=1.0)"]
                  if d in ha_dists]
    if len(show_dists) < 2 and ha_dists:
        show_dists = ha_dists[:2]

    for ax, d in zip(ax_b, show_dists):
        sub = ha[ha["dist"] == d]
        for est in ests:
            s2 = sub[sub["estimator"] == est].sort_values("n")
            ax.errorbar(s2["n"], s2["mean_hat"], yerr=s2["std_hat"],
                        marker=MARKERS[est], color=COLORS[est],
                        linewidth=2, capsize=3, label=LABELS[est])
        ax.set_xscale("log")
        ax.set_xticks(ns); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("n", fontsize=9)
        ax.set_ylabel(r"$\bar{\hat\theta} \pm \hat\sigma$", fontsize=9)
        ax.set_title(f"(B) $\\hat\\theta$ bajo $H_a$: {d}", fontsize=10)
        ax.grid(True, ls="--", alpha=0.4)
    ax_b[0].legend(loc="best", fontsize=8)

    fig.suptitle(
        "Calidad de los estimadores del centro de simetría — $R=500$ réplicas",
        fontsize=13, y=0.98,
    )
    out = outdir / "apendice_calidad_estimadores.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
