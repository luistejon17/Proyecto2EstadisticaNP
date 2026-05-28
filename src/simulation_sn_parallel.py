"""
Versión paralelizada del estudio MC para S_n.

Distribuye las réplicas entre los núcleos disponibles con
ProcessPoolExecutor. Cada tarea es una réplica completa con un seed
único y reproducible.

Compatibilidad Windows: el script llamador debe usar
``if __name__ == '__main__':`` para que el spawn funcione.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

import numpy as np
import pandas as pd

from .characteristic_fn import WeightFn, default_weights, make_t_grid
from .distributions import DistSpec
from .simulation_parallel import _SerializableSpec, _SPEC_REGISTRY, _get_sspec
from .simulation_sn import SimConfigSn  # reutilizamos la config


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _single_replica_sn(
    sspec_name: str,
    under_h0: bool,
    sampler_module: str,
    sampler_name: str,
    sampler_kwargs: dict,
    n: int,
    estimator: str,
    q: int,
    weight_name: str,
    rep: int,
    B: int,
    alpha: float,
    n_t_grid: int,
    seed: int,
) -> dict:
    """Worker: una réplica MC para S_n. Devuelve un dict de resultados."""
    import importlib
    import time as _time

    import numpy as _np

    from src.bootstrap_sn import bootstrap_test_Sn
    from src.characteristic_fn import default_weights as _dw, make_t_grid as _mt

    mod = importlib.import_module(sampler_module)
    factory = getattr(mod, sampler_name)
    spec = factory(**sampler_kwargs)

    weights = _dw()
    w_fn = weights[weight_name]
    t_grid = _mt(w_fn, n_points=n_t_grid)

    rng = _np.random.default_rng(seed)
    sample = spec.sampler(n, rng)
    t0 = _time.perf_counter()
    res = bootstrap_test_Sn(
        sample, w_fn=w_fn, q=q, estimator=estimator,
        B=B, t_grid=t_grid, rng=rng,
    )
    elapsed = _time.perf_counter() - t0

    return dict(
        dist=sspec_name,
        under_h0=under_h0,
        n=n,
        estimator=estimator,
        q=q,
        weight=weight_name,
        rep=rep,
        statistic=res.statistic,
        p_value=res.p_value,
        reject=res.p_value < alpha,
        theta_hat=res.theta_hat,
        time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------
def run_simulation_sn_parallel(
    specs: Iterable[DistSpec],
    config: SimConfigSn,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Versión paralela de `run_simulation_sn`."""
    import os
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 2)

    specs = list(specs)
    sspecs = [_get_sspec(s) for s in specs]

    # Construir lista de tareas
    tasks = []
    task_idx = 0
    for sspec in sspecs:
        for n in config.sample_sizes:
            for est in config.estimators:
                for q in config.qs:
                    for w_name in config.weight_names:
                        for r in range(config.R):
                            seed = config.seed + task_idx
                            tasks.append(
                                (sspec, n, est, q, w_name, r, seed)
                            )
                            task_idx += 1

    total = len(tasks)
    print(
        f"[S_n MC paralelo] {total} tareas en {n_workers} workers "
        f"(B={config.B}, R={config.R})"
    )
    t_start = time.perf_counter()

    rows: list[dict] = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _single_replica_sn,
                sspec.name, sspec.under_h0,
                sspec.module, sspec.factory, sspec.kwargs,
                n, est, q, w_name, r, config.B, config.alpha,
                config.n_t_grid, seed,
            ): (sspec.name, n, est, q, w_name, r)
            for sspec, n, est, q, w_name, r, seed in tasks
        }

        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            pct = int(100 * done / total)
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(
                f"\r[S_n MC paralelo] {pct:3d}%  ({done}/{total})  "
                f"ETA: {eta/60:.1f} min",
                end="", flush=True,
            )

    print(
        f"\n[S_n MC paralelo] Completado en {(time.perf_counter()-t_start)/60:.1f} min"
    )
    return pd.DataFrame(rows)
