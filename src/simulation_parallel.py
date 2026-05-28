"""
Estudio Monte Carlo paralelizado del test bootstrap T_n.

Usa ``concurrent.futures.ProcessPoolExecutor`` para distribuir las R
réplicas entre los núcleos disponibles. Cada réplica es completamente
independiente (distinto seed), así que la paralelización es exacta.

Compatibilidad Windows: el bloque ``if __name__ == '__main__':`` del
script llamador es obligatorio para que el spawn de procesos funcione.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .bootstrap_ks import bootstrap_test_Tn
from .distributions import DistSpec
from .simulation import SimConfig, summarize  # reutilizamos summarize


# ---------------------------------------------------------------------------
# Tarea atómica ejecutable en un worker
# ---------------------------------------------------------------------------
def _single_replica(
    spec_name: str,
    under_h0: bool,
    sampler_module: str,
    sampler_name: str,
    sampler_kwargs: dict,
    n: int,
    estimator: str,
    rep: int,
    B: int,
    alpha: float,
    seed: int,
) -> dict:
    """Worker: una réplica MC. Devuelve un dict con los resultados."""
    import importlib
    import numpy as np

    mod = importlib.import_module(sampler_module)
    factory = getattr(mod, sampler_name)
    spec = factory(**sampler_kwargs)

    rng = np.random.default_rng(seed)
    sample = spec.sampler(n, rng)
    t0 = time.perf_counter()
    from src.bootstrap_ks import bootstrap_test_Tn
    res = bootstrap_test_Tn(sample, estimator=estimator, B=B, rng=rng)
    elapsed = time.perf_counter() - t0

    return dict(
        dist=spec_name,
        under_h0=under_h0,
        n=n,
        estimator=estimator,
        rep=rep,
        statistic=res.statistic,
        p_value=res.p_value,
        reject=res.p_value < alpha,
        theta_hat=res.theta_hat,
        time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Descripción de un escenario de distribución serializable (pickle-safe)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _SerializableSpec:
    """Versión serializable de DistSpec para enviar al worker."""
    name: str
    under_h0: bool
    module: str       # e.g. "src.distributions"
    factory: str      # e.g. "uniform_h0"
    kwargs: dict      # argumentos para la factory


# Catálogo de conversión para las specs por defecto
_SPEC_REGISTRY: dict[str, _SerializableSpec] = {
    "Uniforme(1.0,3.0)": _SerializableSpec(
        "Uniforme(1.0,3.0)", True, "src.distributions", "uniform_h0",
        {"a": 1.0, "b": 3.0}
    ),
    "Cauchy(loc=2.0,scale=1.0)": _SerializableSpec(
        "Cauchy(loc=2.0,scale=1.0)", True, "src.distributions", "cauchy_h0",
        {"loc": 2.0, "scale": 1.0}
    ),
    "Normal(loc=0.0,scale=1.0)": _SerializableSpec(
        "Normal(loc=0.0,scale=1.0)", True, "src.distributions", "normal_h0",
        {"loc": 0.0, "scale": 1.0}
    ),
    "Gamma(k=2.0,s=1.0)": _SerializableSpec(
        "Gamma(k=2.0,s=1.0)", False, "src.distributions", "gamma_ha",
        {"shape": 2.0, "scale": 1.0}
    ),
    "Weibull(k=1.5,s=1.0)": _SerializableSpec(
        "Weibull(k=1.5,s=1.0)", False, "src.distributions", "weibull_ha",
        {"shape": 1.5, "scale": 1.0}
    ),
    "Pareto(a=3.0,s=1.0)": _SerializableSpec(
        "Pareto(a=3.0,s=1.0)", False, "src.distributions", "pareto_ha",
        {"shape": 3.0, "scale": 1.0}
    ),
}


def _get_sspec(spec: DistSpec) -> _SerializableSpec:
    if spec.name not in _SPEC_REGISTRY:
        raise ValueError(
            f"Spec '{spec.name}' no está en el registro. "
            "Agrégala a _SPEC_REGISTRY en simulation_parallel.py."
        )
    return _SPEC_REGISTRY[spec.name]


# ---------------------------------------------------------------------------
# Función principal paralelizada
# ---------------------------------------------------------------------------
def run_simulation_parallel(
    specs: Iterable[DistSpec],
    config: SimConfig,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Versión paralelizada de ``run_simulation``.

    Distribuye las R réplicas MC entre los núcleos disponibles.
    En Windows se requiere que el script llamador tenga
    ``if __name__ == '__main__':`` para que el spawn funcione.

    Parameters
    ----------
    specs : iterable de DistSpec
    config : SimConfig
    n_workers : int o None
        Número de procesos. None → usa ``os.cpu_count()``.

    Returns
    -------
    pd.DataFrame  — misma estructura que ``run_simulation``.
    """
    import os
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 2)

    specs = list(specs)
    sspecs = [_get_sspec(s) for s in specs]

    # Construimos una lista de tareas
    tasks = []
    task_idx = 0
    for sspec in sspecs:
        for n in config.sample_sizes:
            for est in config.estimators:
                for r in range(config.R):
                    # Seed único y reproducible por tarea
                    seed = config.seed + task_idx
                    tasks.append((sspec, n, est, r, seed))
                    task_idx += 1

    total = len(tasks)
    print(f"[T_n MC paralelo] {total} tareas en {n_workers} workers "
          f"(B={config.B}, R={config.R})")
    t_start = time.perf_counter()

    rows: list[dict] = []
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _single_replica,
                sspec.name, sspec.under_h0,
                sspec.module, sspec.factory, sspec.kwargs,
                n, est, r, config.B, config.alpha, seed,
            ): (sspec.name, n, est, r)
            for sspec, n, est, r, seed in tasks
        }

        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            pct = int(100 * done / total)
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(
                f"\r[T_n MC paralelo] {pct:3d}%  ({done}/{total})  "
                f"ETA: {eta/60:.1f} min",
                end="", flush=True,
            )

    print(f"\n[T_n MC paralelo] Completado en {(time.perf_counter()-t_start)/60:.1f} min")
    return pd.DataFrame(rows)
