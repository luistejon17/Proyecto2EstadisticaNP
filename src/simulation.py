"""
Estudio Monte Carlo del test bootstrap T_n.

Para cada combinación (distribución, tamaño n, estimador) se realizan R
réplicas Monte Carlo. En cada réplica se genera una muestra, se ejecuta el
bootstrap y se registra el p-valor, la decisión y el tiempo de ejecución.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .bootstrap_ks import bootstrap_test_Tn
from .distributions import DistSpec


@dataclass
class SimConfig:
    """Configuración del estudio MC para el Test 1 (T_n)."""
    sample_sizes: tuple[int, ...] = (20, 40, 80, 160)
    estimators: tuple[str, ...] = ("argmin", "median", "trimmed")
    B: int = 500
    R: int = 300
    alpha: float = 0.05
    seed: int = 2026
    progress: bool = True
    extra: dict = field(default_factory=dict)


def run_simulation(
    specs: Iterable[DistSpec],
    config: SimConfig,
) -> pd.DataFrame:
    """
    Ejecuta el estudio MC y devuelve un DataFrame con los resultados.

    Cada fila corresponde a una réplica individual. Las columnas son:
        - dist : nombre de la distribución
        - under_h0 : bool
        - n : tamaño muestral
        - estimator : nombre del estimador del centro
        - rep : índice de la réplica
        - statistic : T_n observado
        - p_value : p-valor bootstrap
        - reject : bool al nivel alpha
        - theta_hat : centro estimado
        - time_s : tiempo en segundos del test (incluye B remuestras)
    """
    rng = np.random.default_rng(config.seed)
    rows: list[dict] = []

    specs = list(specs)
    total = len(specs) * len(config.sample_sizes) * len(config.estimators) * config.R
    done = 0
    last_pct = -1

    for spec in specs:
        for n in config.sample_sizes:
            for est in config.estimators:
                for r in range(config.R):
                    sample = spec.sampler(n, rng)
                    t0 = time.perf_counter()
                    res = bootstrap_test_Tn(sample, estimator=est, B=config.B, rng=rng)
                    elapsed = time.perf_counter() - t0
                    rows.append(
                        dict(
                            dist=spec.name,
                            under_h0=spec.under_h0,
                            n=n,
                            estimator=est,
                            rep=r,
                            statistic=res.statistic,
                            p_value=res.p_value,
                            reject=res.p_value < config.alpha,
                            theta_hat=res.theta_hat,
                            time_s=elapsed,
                        )
                    )
                    done += 1
                    if config.progress:
                        pct = int(100 * done / total)
                        if pct != last_pct:
                            print(f"\r[T_n MC] {pct:3d}%  ({done}/{total})", end="", flush=True)
                            last_pct = pct
    if config.progress:
        print()

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Agrega los resultados de ``run_simulation`` por (dist, n, estimator)."""
    g = df.groupby(["dist", "under_h0", "n", "estimator"], as_index=False).agg(
        reject_rate=("reject", "mean"),
        mean_pvalue=("p_value", "mean"),
        mean_stat=("statistic", "mean"),
        mean_time_s=("time_s", "mean"),
        sd_time_s=("time_s", "std"),
        n_rep=("rep", "count"),
    )
    # Banda de fluctuación binomial sqrt(p(1-p)/R) para la tasa de rechazo
    g["se_rate"] = np.sqrt(g["reject_rate"] * (1 - g["reject_rate"]) / g["n_rep"])
    return g
