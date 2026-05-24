"""Gráficas del estudio Monte Carlo para S_n (y comparativas T_n vs S_n)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gráficas básicas por (q, w)
# ---------------------------------------------------------------------------
def plot_type_i_error_sn(
    summary: pd.DataFrame,
    alpha: float,
    outdir: Path,
) -> list[Path]:
    """Error Tipo I vs n, una sub-gráfica por distribución H0, una figura por (q, w)."""
    h0 = summary[summary["under_h0"]].copy()
    if h0.empty:
        return []
    out_paths: list[Path] = []
    for (q, w), sub_qw in h0.groupby(["q", "weight"]):
        dists = sorted(sub_qw["dist"].unique())
        estimators = sorted(sub_qw["estimator"].unique())
        fig, axes = plt.subplots(
            1, len(dists), figsize=(5.5 * len(dists), 4.2),
            sharey=True, squeeze=False,
        )
        for ax, d in zip(axes[0], dists):
            s_d = sub_qw[sub_qw["dist"] == d]
            for est in estimators:
                s2 = s_d[s_d["estimator"] == est].sort_values("n")
                if s2.empty:
                    continue
                ax.errorbar(
                    s2["n"], s2["reject_rate"], yerr=2 * s2["se_rate"],
                    marker="o", capsize=3, label=est,
                )
            ax.axhline(alpha, color="black", linestyle="--", linewidth=1,
                       label=f"α={alpha}")
            ax.set_xscale("log")
            ax.set_xlabel("n (log)")
            ax.set_title(d)
            ax.grid(True, alpha=0.3)
        axes[0, 0].set_ylabel("Tasa de rechazo (Error Tipo I)")
        axes[0, -1].legend(loc="best", fontsize=8)
        fig.suptitle(f"Error Tipo I — S_n  (q={q}, w={w})", y=1.02)
        fig.tight_layout()
        out = outdir / f"sn_type_i_error_q{q}_{_safe(w)}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def plot_power_curves_sn(
    summary: pd.DataFrame,
    outdir: Path,
) -> list[Path]:
    """Potencia vs n, una sub-gráfica por distribución Ha, una figura por (q, w)."""
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty:
        return []
    out_paths: list[Path] = []
    for (q, w), sub_qw in ha.groupby(["q", "weight"]):
        dists = sorted(sub_qw["dist"].unique())
        estimators = sorted(sub_qw["estimator"].unique())
        n_cols = min(3, len(dists))
        n_rows = int(np.ceil(len(dists) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows),
            sharey=True, squeeze=False,
        )
        axes_flat = axes.flatten()
        for ax, d in zip(axes_flat, dists):
            s_d = sub_qw[sub_qw["dist"] == d]
            for est in estimators:
                s2 = s_d[s_d["estimator"] == est].sort_values("n")
                if s2.empty:
                    continue
                ax.errorbar(
                    s2["n"], s2["reject_rate"], yerr=2 * s2["se_rate"],
                    marker="o", capsize=3, label=est,
                )
            ax.set_xscale("log")
            ax.set_xlabel("n (log)")
            ax.set_title(d)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[len(dists):]:
            ax.set_visible(False)
        axes[0, 0].set_ylabel("Potencia empírica")
        axes_flat[0].legend(loc="best", fontsize=8)
        fig.suptitle(f"Potencia — S_n  (q={q}, w={w})", y=1.02)
        fig.tight_layout()
        out = outdir / f"sn_power_curves_q{q}_{_safe(w)}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


# ---------------------------------------------------------------------------
# Comparativas: q=1 vs q=2 y distintas w(t)
# ---------------------------------------------------------------------------
def plot_q1_vs_q2(summary: pd.DataFrame, outdir: Path) -> Path | None:
    """Compara potencia entre q=1 y q=2 (estimador argmin, peso por defecto).

    Una sub-gráfica por distribución Ha, una curva por q. Promediado
    sobre los pesos disponibles.
    """
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty or ha["q"].nunique() < 2:
        return None
    sub = ha[ha["estimator"] == "argmin"].copy()
    if sub.empty:
        sub = ha.copy()
    # Promediar sobre pesos
    g = sub.groupby(["dist", "n", "q"], as_index=False).agg(
        reject_rate=("reject_rate", "mean"),
    )
    dists = sorted(g["dist"].unique())
    n_cols = min(3, len(dists))
    n_rows = int(np.ceil(len(dists) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    for ax, d in zip(axes_flat, dists):
        for q in sorted(g["q"].unique()):
            s2 = g[(g["dist"] == d) & (g["q"] == q)].sort_values("n")
            ax.plot(s2["n"], s2["reject_rate"], marker="o", label=f"q={q}")
        ax.set_xscale("log")
        ax.set_xlabel("n (log)")
        ax.set_title(d)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(dists):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Potencia empírica (promedio sobre pesos)")
    axes_flat[0].legend(loc="best", fontsize=9)
    fig.suptitle("S_n: comparación q=1 vs q=2 (estimador argmin)", y=1.02)
    fig.tight_layout()
    out = outdir / "sn_compare_q.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_weights_compare(summary: pd.DataFrame, outdir: Path) -> Path | None:
    """Compara potencia entre distintas funciones de peso w(t).

    Estimador = argmin, q = 2. Una sub-gráfica por distribución Ha,
    una curva por peso.
    """
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty or ha["weight"].nunique() < 2:
        return None
    sub = ha[(ha["estimator"] == "argmin") & (ha["q"] == 2)].copy()
    if sub.empty:
        return None
    dists = sorted(sub["dist"].unique())
    n_cols = min(3, len(dists))
    n_rows = int(np.ceil(len(dists) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    for ax, d in zip(axes_flat, dists):
        for w in sorted(sub["weight"].unique()):
            s2 = sub[(sub["dist"] == d) & (sub["weight"] == w)].sort_values("n")
            if s2.empty:
                continue
            ax.errorbar(
                s2["n"], s2["reject_rate"], yerr=2 * s2["se_rate"],
                marker="o", capsize=3, label=w,
            )
        ax.set_xscale("log")
        ax.set_xlabel("n (log)")
        ax.set_title(d)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(dists):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Potencia empírica")
    axes_flat[0].legend(loc="best", fontsize=8, title="w(t)")
    fig.suptitle("S_n: comparación de funciones de peso (argmin, q=2)", y=1.02)
    fig.tight_layout()
    out = outdir / "sn_compare_weights.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Heatmap de potencia
# ---------------------------------------------------------------------------
def plot_power_heatmap(summary: pd.DataFrame, outdir: Path) -> Path | None:
    """Heatmap de potencia con (dist, n) en filas y (weight, q) en columnas.

    Estimador = argmin.
    """
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty:
        return None
    sub = ha[ha["estimator"] == "argmin"].copy()
    if sub.empty:
        return None

    # Construimos etiquetas combinadas
    sub["row"] = sub["dist"] + " | n=" + sub["n"].astype(str)
    sub["col"] = "q=" + sub["q"].astype(str) + " | " + sub["weight"]
    pivot = sub.pivot_table(index="row", columns="col", values="reject_rate")
    # Ordenamos filas por dist, n
    sub_order = sub.drop_duplicates("row").sort_values(["dist", "n"])
    pivot = pivot.reindex(sub_order["row"])

    fig, ax = plt.subplots(
        figsize=(0.8 * len(pivot.columns) + 3, 0.35 * len(pivot.index) + 1.5)
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Potencia empírica")
    ax.set_title("Heatmap de potencia — S_n (estimador argmin)")
    fig.tight_layout()
    out = outdir / "sn_power_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Tiempos
# ---------------------------------------------------------------------------
def plot_runtime_sn(summary: pd.DataFrame, outdir: Path) -> Path:
    """Tiempo medio por test vs n, una curva por estimador (promedio sobre dists)."""
    df = summary.copy()
    g = df.groupby(["n", "estimator"], as_index=False).agg(
        mean_time_s=("mean_time_s", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for est in sorted(g["estimator"].unique()):
        s2 = g[g["estimator"] == est].sort_values("n")
        ax.plot(s2["n"], s2["mean_time_s"], marker="o", label=est)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log)")
    ax.set_ylabel("Tiempo medio por test [s] (log)")
    ax.set_title("Costo computacional del test S_n (promedio sobre q, w)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = outdir / "sn_runtime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pvalue_distribution_h0_sn(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Histograma del p-valor bajo H0 (debería ser ~uniforme)."""
    h0 = df[df["under_h0"]].copy()
    if h0.empty:
        return None
    # Fijamos argmin y q=2 para no saturar la figura
    sub = h0[(h0["estimator"] == "argmin") & (h0["q"] == 2)].copy()
    if sub.empty:
        sub = h0.copy()
    dists = sorted(sub["dist"].unique())
    weights = sorted(sub["weight"].unique())
    fig, axes = plt.subplots(
        len(dists), len(weights),
        figsize=(3.2 * len(weights), 2.7 * len(dists)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, d in enumerate(dists):
        for j, w in enumerate(weights):
            ax = axes[i, j]
            s2 = sub[(sub["dist"] == d) & (sub["weight"] == w)]
            ax.hist(s2["p_value"], bins=20, range=(0, 1),
                    color="C0", edgecolor="white", density=True)
            ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{d} | {w}", fontsize=8)
            ax.set_xlim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("p-valor")
    for ax in axes[:, 0]:
        ax.set_ylabel("densidad")
    fig.suptitle("Distribución del p-valor bajo H0 — S_n (argmin, q=2)", y=1.02)
    fig.tight_layout()
    out = outdir / "sn_pvalue_h0.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Comparación T_n vs S_n
# ---------------------------------------------------------------------------
def plot_tn_vs_sn_power(
    summary_tn: pd.DataFrame,
    summary_sn: pd.DataFrame,
    outdir: Path,
) -> Path | None:
    """Compara la potencia de T_n vs la mejor configuración de S_n por dist.

    Para S_n usamos argmin con la (q, w) que maximiza potencia promediada
    sobre n. Para T_n usamos argmin.
    """
    ha_tn = summary_tn[~summary_tn["under_h0"]].copy()
    ha_sn = summary_sn[~summary_sn["under_h0"]].copy()
    if ha_tn.empty or ha_sn.empty:
        return None

    tn = ha_tn[ha_tn["estimator"] == "argmin"].copy()
    sn = ha_sn[ha_sn["estimator"] == "argmin"].copy()
    if tn.empty or sn.empty:
        return None

    # Para cada dist en Sn, elegir la (q, w) con mejor potencia media
    sn_best_qw = (
        sn.groupby(["dist", "q", "weight"], as_index=False)["reject_rate"]
          .mean()
          .sort_values("reject_rate", ascending=False)
          .drop_duplicates("dist")
    )

    dists = sorted(set(tn["dist"]).intersection(sn["dist"]))
    n_cols = min(3, len(dists))
    n_rows = int(np.ceil(len(dists) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.8 * n_cols, 3.6 * n_rows),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()
    for ax, d in zip(axes_flat, dists):
        # T_n
        s_tn = tn[tn["dist"] == d].sort_values("n")
        if not s_tn.empty:
            ax.errorbar(
                s_tn["n"], s_tn["reject_rate"], yerr=2 * s_tn["se_rate"],
                marker="o", capsize=3, label="T_n (argmin)", color="C0",
            )
        # S_n best
        best_row = sn_best_qw[sn_best_qw["dist"] == d]
        if not best_row.empty:
            best_q = int(best_row.iloc[0]["q"])
            best_w = best_row.iloc[0]["weight"]
            s_sn = sn[
                (sn["dist"] == d) & (sn["q"] == best_q) & (sn["weight"] == best_w)
            ].sort_values("n")
            ax.errorbar(
                s_sn["n"], s_sn["reject_rate"], yerr=2 * s_sn["se_rate"],
                marker="s", capsize=3,
                label=f"S_n best (q={best_q}, {best_w})", color="C1",
            )
        ax.set_xscale("log")
        ax.set_xlabel("n (log)")
        ax.set_title(d)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(dists):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Potencia empírica")
    axes_flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("Comparación T_n vs S_n (mejor (q, w))", y=1.02)
    fig.tight_layout()
    out = outdir / "compare_tn_vs_sn_power.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tn_vs_sn_runtime(
    summary_tn: pd.DataFrame,
    summary_sn: pd.DataFrame,
    outdir: Path,
) -> Path:
    """Tiempo medio por test: T_n vs S_n por estimador."""
    tn = summary_tn.copy()
    sn = summary_sn.copy()
    tn_g = tn.groupby(["n", "estimator"], as_index=False).agg(
        mean_time_s=("mean_time_s", "mean"),
    )
    sn_g = sn.groupby(["n", "estimator"], as_index=False).agg(
        mean_time_s=("mean_time_s", "mean"),
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {"argmin": "C0", "median": "C1", "trimmed": "C2"}
    for est in sorted(set(tn_g["estimator"]).intersection(sn_g["estimator"])):
        col = colors.get(est, "k")
        s1 = tn_g[tn_g["estimator"] == est].sort_values("n")
        s2 = sn_g[sn_g["estimator"] == est].sort_values("n")
        ax.plot(s1["n"], s1["mean_time_s"], marker="o", linestyle="-",
                color=col, label=f"T_n | {est}")
        ax.plot(s2["n"], s2["mean_time_s"], marker="s", linestyle="--",
                color=col, label=f"S_n | {est}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log)")
    ax.set_ylabel("Tiempo medio por test [s] (log)")
    ax.set_title("Costo computacional: T_n vs S_n")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = outdir / "compare_tn_vs_sn_runtime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Utilidad
# ---------------------------------------------------------------------------
def _safe(s: str) -> str:
    return (
        s.replace("(", "_").replace(")", "")
        .replace(",", "_").replace("=", "")
        .replace(" ", "").replace(".", "p")
    )
