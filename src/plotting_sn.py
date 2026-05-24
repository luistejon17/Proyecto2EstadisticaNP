"""Gráficas del estudio Monte Carlo para S_n (y comparativas T_n vs S_n)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap


# ---------------------------------------------------------------------------
# Utilidad interna
# ---------------------------------------------------------------------------
def _safe(s: str) -> str:
    return (
        s.replace("(", "_").replace(")", "")
        .replace(",", "_").replace("=", "")
        .replace(" ", "").replace(".", "p")
    )


def _sigma_band_heatmap(ax, pivot: pd.DataFrame, R: int, alpha: float = 0.05,
                         title: str = ""):
    """Heatmap de Error Tipo I con colormap discreto por bandas σ."""
    mu = alpha * 100
    sigma = np.sqrt(alpha * (1 - alpha) / R) * 100
    limites = [0, mu - 3*sigma, mu - 2*sigma, mu - sigma,
               mu + sigma, mu + 2*sigma, mu + 3*sigma, 100]
    colores = ["darkred", "tomato", "khaki", "seagreen", "khaki", "tomato", "darkred"]
    cmap_d = ListedColormap(colores)
    norm = BoundaryNorm(limites, cmap_d.N)
    tick_pos = [(limites[i] + limites[i+1]) / 2 for i in range(len(limites)-1)]
    tick_lbl = [
        r"$>3\sigma$ (muy bajo)", r"$2\sigma$–$3\sigma$ (bajo)",
        r"$1\sigma$–$2\sigma$", r"$<1\sigma$ (diana)",
        r"$1\sigma$–$2\sigma$", r"$2\sigma$–$3\sigma$ (alto)",
        r"$>3\sigma$ (muy alto)",
    ]
    sns.set_style("white")
    hm = sns.heatmap(
        pivot * 100, annot=True, fmt=".1f",
        cmap=cmap_d, norm=norm,
        linewidths=0.5, linecolor="white",
        cbar_kws={"ticks": tick_pos, "label": f"Evaluación respecto al {int(alpha*100)}% teórico"},
        ax=ax,
    )
    hm.collections[0].colorbar.set_ticklabels(tick_lbl)
    ax.set_title(title, pad=12)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)


# ---------------------------------------------------------------------------
# Error Tipo I — líneas por (q, w)
# ---------------------------------------------------------------------------
def plot_type_i_error_sn(
    summary: pd.DataFrame,
    alpha: float,
    outdir: Path,
) -> list[Path]:
    """Error Tipo I vs n, una figura por (q, w)."""
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
                    marker="o", capsize=3, linewidth=2, label=est,
                )
            ax.axhline(alpha, color="black", linestyle="--", linewidth=1,
                       label=f"α={alpha}")
            ax.set_xscale("log")
            ax.set_xlabel("n (log)")
            ax.set_title(d)
            ax.grid(True, linestyle="--", alpha=0.5)
        axes[0, 0].set_ylabel("Tasa de rechazo (Error Tipo I)")
        axes[0, -1].legend(loc="best", fontsize=8)
        fig.suptitle(f"Error Tipo I — S_n  (q={q}, w={w})", y=1.02)
        fig.tight_layout()
        out = outdir / f"sn_type_i_error_q{q}_{_safe(w)}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


# ---------------------------------------------------------------------------
# Error Tipo I — heatmap sigma-band
# ---------------------------------------------------------------------------
def plot_type_i_error_heatmap_sn(
    summary: pd.DataFrame,
    alpha: float,
    outdir: Path,
    R: int = 200,
) -> list[Path]:
    """Heatmap de Error Tipo I con bandas σ, una figura por (q, w)."""
    h0 = summary[summary["under_h0"]].copy()
    if h0.empty:
        return []
    out_paths: list[Path] = []
    for (q, w), sub_qw in h0.groupby(["q", "weight"]):
        pivot = sub_qw.pivot_table(
            index=["dist", "n"], columns="estimator", values="reject_rate"
        )
        pivot.index = [f"{d} | n={n}" for d, n in pivot.index]
        fig, ax = plt.subplots(
            figsize=(max(6, 2.2 * len(pivot.columns)),
                     max(4, 0.45 * len(pivot.index) + 1.5))
        )
        _sigma_band_heatmap(
            ax, pivot, R=R, alpha=alpha,
            title=f"Error Tipo I (%) — S_n  (q={q}, w={w})",
        )
        fig.tight_layout()
        out = outdir / f"sn_type_i_heatmap_q{q}_{_safe(w)}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


# ---------------------------------------------------------------------------
# Potencia — líneas por (q, w)
# ---------------------------------------------------------------------------
def plot_power_curves_sn(
    summary: pd.DataFrame,
    outdir: Path,
) -> list[Path]:
    """Potencia vs n, una figura por (q, w)."""
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
                    marker="o", capsize=3, linewidth=2, label=est,
                )
            ax.set_xscale("log")
            ax.set_xlabel("n (log)")
            ax.set_title(d)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, linestyle="--", alpha=0.5)
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
# Potencia — heatmap RdYlGn (argmin)
# ---------------------------------------------------------------------------
def plot_power_heatmap(summary: pd.DataFrame, outdir: Path) -> Path | None:
    """Heatmap de potencia RdYlGn con filas=(dist, n), columnas=(q, w).

    Estimador = argmin (el más informativo).
    """
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty:
        return None
    sub = ha[ha["estimator"] == "argmin"].copy()
    if sub.empty:
        return None

    sub["col"] = "q=" + sub["q"].astype(str) + " | " + sub["weight"]
    pivot = sub.pivot_table(index=["dist", "n"], columns="col",
                            values="reject_rate") * 100
    pivot.index = [f"{d} | n={n}" for d, n in pivot.index]
    # Ordenar columnas
    pivot = pivot[sorted(pivot.columns)]

    fig, ax = plt.subplots(
        figsize=(max(8, 1.5 * len(pivot.columns) + 2),
                 max(5, 0.42 * len(pivot.index) + 1.5))
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
    ax.set_title("Potencia empírica (%) — S_n  (estimador argmin, mayor es mejor)", pad=12)
    ax.set_xlabel("(q, función de peso)", fontsize=10)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    out = outdir / "sn_power_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Comparativas: q=1 vs q=2
# ---------------------------------------------------------------------------
def plot_q1_vs_q2(summary: pd.DataFrame, outdir: Path) -> Path | None:
    """Compara potencia entre q=1 y q=2 (argmin, promediado sobre pesos)."""
    ha = summary[~summary["under_h0"]].copy()
    if ha.empty or ha["q"].nunique() < 2:
        return None
    sub = ha[ha["estimator"] == "argmin"].copy()
    if sub.empty:
        sub = ha.copy()
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
            ax.plot(s2["n"], s2["reject_rate"], marker="o", linewidth=2,
                    label=f"q={q}")
        ax.set_xscale("log")
        ax.set_xlabel("n (log)")
        ax.set_title(d)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
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


# ---------------------------------------------------------------------------
# Comparativas: funciones de peso
# ---------------------------------------------------------------------------
def plot_weights_compare(summary: pd.DataFrame, outdir: Path) -> Path | None:
    """Compara potencia entre distintas w(t) (argmin, q=2)."""
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
                marker="o", capsize=3, linewidth=2, label=w,
            )
        ax.set_xscale("log")
        ax.set_xlabel("n (log)")
        ax.set_title(d)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
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
# Runtime
# ---------------------------------------------------------------------------
def plot_runtime_sn(summary: pd.DataFrame, outdir: Path) -> Path:
    """Tiempo medio por test vs n, una curva por estimador."""
    df = summary.copy()
    g = df.groupby(["n", "estimator"], as_index=False).agg(
        mean_time_s=("mean_time_s", "mean"),
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for est in sorted(g["estimator"].unique()):
        s2 = g[g["estimator"] == est].sort_values("n")
        ax.plot(s2["n"], s2["mean_time_s"], marker="o", linewidth=2, label=est)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log)")
    ax.set_ylabel("Tiempo medio por test [s] (log)")
    ax.set_title("Costo computacional del test S_n (promedio sobre q, w)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = outdir / "sn_runtime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# p-valor bajo H0
# ---------------------------------------------------------------------------
def plot_pvalue_distribution_h0_sn(df: pd.DataFrame, outdir: Path) -> Path | None:
    """Histograma del p-valor bajo H0 (debería ser ~uniforme)."""
    h0 = df[df["under_h0"]].copy()
    if h0.empty:
        return None
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
            ax.grid(True, linestyle="--", alpha=0.4)
    for ax in axes[-1]:
        ax.set_xlabel("p-valor")
    for ax in axes[:, 0]:
        ax.set_ylabel("densidad")
    fig.suptitle("Distribución del p-valor bajo H_0 — S_n (argmin, q=2)", y=1.02)
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
    """Potencia T_n vs mejor configuración de S_n, por distribución."""
    ha_tn = summary_tn[~summary_tn["under_h0"]].copy()
    ha_sn = summary_sn[~summary_sn["under_h0"]].copy()
    if ha_tn.empty or ha_sn.empty:
        return None
    tn = ha_tn[ha_tn["estimator"] == "argmin"].copy()
    sn = ha_sn[ha_sn["estimator"] == "argmin"].copy()
    if tn.empty or sn.empty:
        return None

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
        s_tn = tn[tn["dist"] == d].sort_values("n")
        if not s_tn.empty:
            ax.errorbar(s_tn["n"], s_tn["reject_rate"], yerr=2 * s_tn["se_rate"],
                        marker="o", capsize=3, linewidth=2,
                        label="T_n (argmin)", color="C0")
        best_row = sn_best_qw[sn_best_qw["dist"] == d]
        if not best_row.empty:
            bq = int(best_row.iloc[0]["q"])
            bw = best_row.iloc[0]["weight"]
            s_sn = sn[(sn["dist"] == d) & (sn["q"] == bq) & (sn["weight"] == bw)
                      ].sort_values("n")
            ax.errorbar(s_sn["n"], s_sn["reject_rate"], yerr=2 * s_sn["se_rate"],
                        marker="s", capsize=3, linewidth=2,
                        label=f"S_n best (q={bq}, {bw})", color="C1")
        ax.set_xscale("log")
        ax.set_xlabel("n (log)")
        ax.set_title(d)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)
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
    tn_g = summary_tn.groupby(["n", "estimator"], as_index=False).agg(
        mean_time_s=("mean_time_s", "mean"))
    sn_g = summary_sn.groupby(["n", "estimator"], as_index=False).agg(
        mean_time_s=("mean_time_s", "mean"))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {"argmin": "C0", "median": "C1", "trimmed": "C2"}
    for est in sorted(set(tn_g["estimator"]).intersection(sn_g["estimator"])):
        col = colors.get(est, "k")
        s1 = tn_g[tn_g["estimator"] == est].sort_values("n")
        s2 = sn_g[sn_g["estimator"] == est].sort_values("n")
        ax.plot(s1["n"], s1["mean_time_s"], marker="o", linestyle="-",
                linewidth=2, color=col, label=f"T_n | {est}")
        ax.plot(s2["n"], s2["mean_time_s"], marker="s", linestyle="--",
                linewidth=2, color=col, label=f"S_n | {est}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (log)")
    ax.set_ylabel("Tiempo medio por test [s] (log)")
    ax.set_title("Costo computacional: T_n vs S_n")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = outdir / "compare_tn_vs_sn_runtime.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
