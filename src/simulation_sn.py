"""
Estudio Monte Carlo del test bootstrap S_n.

Para cada combinación (distribución, n, estimador, q, peso) se realizan R
réplicas. Cada fila del DataFrame de salida corresponde a una réplica.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .bootstrap_sn import bootstrap_test_Sn
from .characteristic_fn import WeightFn, default_weights, make_t_grid
from .distributions import DistSpec


@dataclass
class SimConfigSn:
    """Configuración del estudio MC para el Test 2 (S_n)."""
    sample_sizes: tuple[int, ...] = (20, 40, 80, 160)
    estimators: tuple[str, ...] = ("argmin", "median", "trimmed")
    qs: tuple[int, ...] = (1, 2)
    weight_names: tuple[str, ...] = ("gauss_1.0", "gauss_0.5", "laplace_1.0")
    B: int = 500
    R: int = 300
    alpha: float = 0.05
    seed: int = 2026
    n_t_grid: int = 301
    progress: bool = True
    extra: dict = field(default_factory=dict)


def run_simulation_sn(
    specs: Iterable[DistSpec],
    config: SimConfigSn,
    weights_dict: dict[str, WeightFn] | None = None,
) -> pd.DataFrame:
    """Ejecuta el estudio MC secuencial para S_n.

    Columnas del DataFrame de salida:
        dist, under_h0, n, estimator, q, weight, rep,
        statistic, p_value, reject, theta_hat, time_s.
    """
    if weights_dict is None:
        weights_dict = default_weights()

    rng = np.random.default_rng(config.seed)
    rows: list[dict] = []

    specs = list(specs)
    total = (
        len(specs)
        * len(config.sample_sizes)
        * len(config.estimators)
        * len(config.qs)
        * len(config.weight_names)
        * config.R
    )
    done = 0
    last_pct = -1

    for w_name in config.weight_names:
        w_fn = weights_dict[w_name]
        t_grid = make_t_grid(w_fn, n_points=config.n_t_grid)
        for q in config.qs:
            for spec in specs:
                for n in config.sample_sizes:
                    for est in config.estimators:
                        for r in range(config.R):
                            sample = spec.sampler(n, rng)
                            t0 = time.perf_counter()
                            res = bootstrap_test_Sn(
                                sample,
                                w_fn=w_fn,
                                q=q,
                                estimator=est,
                                B=config.B,
                                t_grid=t_grid,
                                rng=rng,
                            )
                            elapsed = time.perf_counter() - t0
                            rows.append(
                                dict(
                                    dist=spec.name,
                                    under_h0=spec.under_h0,
                                    n=n,
                                    estimator=est,
                                    q=q,
                                    weight=w_name,
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
                                    print(
                                        f"\r[S_n MC] {pct:3d}%  ({done}/{total})",
                                        end="", flush=True,
                                    )
                                    last_pct = pct
    if config.progress:
        print()
    return pd.DataFrame(rows)


def summarize_sn(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Agrega los resultados por (dist, n, estimador, q, peso)."""
    g = df.groupby(
        ["dist", "under_h0", "n", "estimator", "q", "weight"], as_index=False
    ).agg(
        reject_rate=("reject", "mean"),
        mean_pvalue=("p_value", "mean"),
        mean_stat=("statistic", "mean"),
        mean_time_s=("time_s", "mean"),
        sd_time_s=("time_s", "std"),
        n_rep=("rep", "count"),
    )
    g["se_rate"] = np.sqrt(g["reject_rate"] * (1 - g["reject_rate"]) / g["n_rep"])
    return g
