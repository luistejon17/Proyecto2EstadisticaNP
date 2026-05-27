"""
Corrida rapida para probar S_n con q=1 y estimador argmin_pyomo.

Por defecto usa una configuracion pequena para validar que Pyomo, el solver y
el pipeline bootstrap funcionan. Con --full aumenta un poco la corrida, pero
sigue siendo una prueba acotada; no reemplaza la simulacion final del proyecto.

Uso:
    python scripts/run_sn_q1_pyomo_test.py
    python scripts/run_sn_q1_pyomo_test.py --full
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _check_pyomo_ready() -> list[str]:
    import pyomo.environ as pyo

    from src.pyomo_argmin import DEFAULT_PYOMO_SOLVERS

    available = [
        solver
        for solver in DEFAULT_PYOMO_SOLVERS
        if pyo.SolverFactory(solver).available(exception_flag=False)
    ]
    if not available:
        raise RuntimeError(
            "Pyomo esta instalado, pero no hay solver disponible. "
            "Instala highspy, GLPK o CBC."
        )
    return available


def main(full: bool = False) -> None:
    from src.distributions import cauchy_h0, gamma_ha, uniform_h0
    from src.simulation_sn import SimConfigSn, summarize_sn
    from src.simulation_sn_parallel import run_simulation_sn_parallel

    solvers = _check_pyomo_ready()
    print(f"Solvers Pyomo disponibles: {solvers}")

    data_dir = ROOT / "results" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        uniform_h0(1.0, 3.0),
        cauchy_h0(2.0, 1.0),
        gamma_ha(2.0, 1.0),
    ]

    if full:
        config = SimConfigSn(
            sample_sizes=(20, 40),
            estimators=("argmin_pyomo",),
            qs=(1,),
            weight_names=("gauss_1.0", "laplace_1.0"),
            B=39,
            R=10,
            alpha=0.05,
            seed=2026,
            n_t_grid=101,
        )
        suffix = ""
    else:
        config = SimConfigSn(
            sample_sizes=(20,),
            estimators=("argmin_pyomo",),
            qs=(1,),
            weight_names=("gauss_1.0",),
            B=19,
            R=3,
            alpha=0.05,
            seed=2026,
            n_t_grid=61,
        )
        suffix = "_quick"

    n_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
    total = (
        len(specs)
        * len(config.sample_sizes)
        * len(config.estimators)
        * len(config.qs)
        * len(config.weight_names)
        * config.R
    )
    print(
        "S_n q=1 con argmin_pyomo: "
        f"{total} tareas, B={config.B}, R={config.R}, "
        f"K={config.n_t_grid}, workers={n_workers}"
    )

    t0 = time.perf_counter()
    df = run_simulation_sn_parallel(specs, config, n_workers=n_workers)
    summary = summarize_sn(df, alpha=config.alpha)
    elapsed = time.perf_counter() - t0

    raw_csv = data_dir / f"sn_pyomo_q1_raw{suffix}.csv"
    summary_csv = data_dir / f"sn_pyomo_q1_summary{suffix}.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    print(f"\nCompletado en {elapsed/60:.2f} min")
    print(f"Guardado:\n  {raw_csv}\n  {summary_csv}")
    print("\nResumen tasas de rechazo:")
    print(
        summary.pivot_table(
            index=["dist", "estimator", "q", "weight"],
            columns="n",
            values="reject_rate",
        )
        .round(3)
        .to_string()
    )
    print("\nTiempos medios por replica:")
    print(
        summary.pivot_table(
            index=["dist", "estimator", "q", "weight"],
            columns="n",
            values="mean_time_s",
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main(full="--full" in sys.argv)
