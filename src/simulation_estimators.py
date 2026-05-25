"""
Simulación Monte Carlo para evaluar la calidad de los estimadores del centro
de simetría como estimadores puntuales (bias, RMSE, MAE).

Se evalúan 4 estimadores:
  - argmin   : Schuster-Narvarte exacto (minimiza T_n)
  - HL       : Hodges-Lehmann (mediana de Walsh averages)
  - median   : mediana muestral
  - trimmed  : media afeitada 10%

Bajo H_0 (distribuciones simétricas con θ_true=2): se reportan sesgo, RMSE y MAE.
Bajo H_a (distribuciones asimétricas): se reporta media y std de θ̂ (no existe θ_true).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .statistics_ks import (
    theta_argmin_schuster_narvarte,
    theta_hodges_lehmann,
    theta_median,
    theta_trimmed,
)
from .distributions import (
    uniform_h0, cauchy_h0,
    gamma_ha, weibull_ha, pareto_ha,
)


ESTIMATORS_QUALITY = {
    "argmin":  theta_argmin_schuster_narvarte,
    "HL":      theta_hodges_lehmann,
    "median":  theta_median,
    "trimmed": theta_trimmed,
}

ESTIMATOR_LABELS = {
    "argmin":  r"$\hat\theta_{\min}$ (argmin $T_n$)",
    "HL":      r"$\hat\theta_{HL}$ (Hodges-Lehmann)",
    "median":  r"$\hat\theta_{\mathrm{med}}$ (mediana)",
    "trimmed": r"$\hat\theta_{\alpha}$ (media afeitada)",
}


def run_quality_simulation(
    sample_sizes: tuple[int, ...] = (20, 40, 80, 160),
    R: int = 500,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Ejecuta R réplicas por (distribución × n × estimador) y retorna un
    DataFrame con las estimaciones de theta.

    Columnas: dist, under_h0, theta_true, n, estimator, rep, theta_hat
    """
    rng = np.random.default_rng(seed)
    specs = [
        uniform_h0(1.0, 3.0),
        cauchy_h0(2.0, 1.0),
        gamma_ha(2.0, 1.0),
        weibull_ha(1.5, 1.0),
        pareto_ha(3.0, 1.0),
    ]
    # Para H0, theta_true = 2 (centro de simetría conocido).
    # Para Ha no existe theta_true; se usa NaN.
    theta_trues = {True: 2.0, False: float("nan")}

    rows = []
    total = len(specs) * len(sample_sizes) * len(ESTIMATORS_QUALITY) * R
    done = 0
    t0 = time.perf_counter()

    for spec in specs:
        for n in sample_sizes:
            for est_name, est_fn in ESTIMATORS_QUALITY.items():
                for r in range(R):
                    x = spec.sampler(n, rng)
                    th = est_fn(x)
                    rows.append({
                        "dist": spec.name,
                        "under_h0": spec.under_h0,
                        "theta_true": theta_trues[spec.under_h0],
                        "n": n,
                        "estimator": est_name,
                        "rep": r,
                        "theta_hat": th,
                    })
                    done += 1
                elapsed = time.perf_counter() - t0
                eta = elapsed / done * (total - done) / 60 if done > 0 else 0
                print(f"\r  {done}/{total} ({100*done/total:.1f}%)  ETA: {eta:.1f} min",
                      end="", flush=True)

    print()
    return pd.DataFrame(rows)


def summarize_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula sesgo, RMSE, MAE y std de theta_hat por (dist, n, estimator).

    Para distribuciones H0 (theta_true=2): bias, RMSE, MAE están definidos.
    Para distribuciones Ha: bias/RMSE/MAE son NaN; se reportan mean y std.
    """
    rows = []
    for (dist, under_h0, theta_true, n, est), g in df.groupby(
        ["dist", "under_h0", "theta_true", "n", "estimator"]
    ):
        th = g["theta_hat"].values
        mean_th = float(np.mean(th))
        std_th = float(np.std(th, ddof=1))
        if under_h0:
            bias = mean_th - theta_true
            rmse = float(np.sqrt(np.mean((th - theta_true) ** 2)))
            mae  = float(np.mean(np.abs(th - theta_true)))
        else:
            bias = rmse = mae = float("nan")
        rows.append({
            "dist": dist,
            "under_h0": under_h0,
            "theta_true": theta_true,
            "n": n,
            "estimator": est,
            "mean_hat": mean_th,
            "std_hat": std_th,
            "bias": bias,
            "rmse": rmse,
            "mae": mae,
        })
    return pd.DataFrame(rows)
