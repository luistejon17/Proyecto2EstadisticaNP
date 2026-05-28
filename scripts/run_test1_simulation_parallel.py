"""
Script: estudio Monte Carlo paralelizado del Test 1 (T_n, Schuster-Barker).

Ejecución:
    python scripts/run_test1_simulation_parallel.py           # run full
    python scripts/run_test1_simulation_parallel.py --quick   # run corto

IMPORTANTE (Windows): este bloque if __name__ == '__main__' es obligatorio
para que ProcessPoolExecutor pueda hacer spawn de los workers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(quick: bool = False) -> None:
    import os
    from src.distributions import default_h0_specs, default_ha_specs
    from src.plotting import (
        plot_power_curves,
        plot_power_vs_cost,
        plot_pvalue_distribution_h0,
        plot_runtime,
        plot_type_i_error,
    )
    from src.simulation import SimConfig, summarize
    from src.simulation_parallel import run_simulation_parallel

    data_dir = ROOT / "results" / "data"
    fig_dir = ROOT / "results" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, os.cpu_count() or 2)
    print(f"CPU cores disponibles: {os.cpu_count()}  |  Workers a usar: {n_workers}")

    if quick:
        config = SimConfig(
            sample_sizes=(20, 40, 80),
            estimators=("argmin", "median", "trimmed"),
            B=199,
            R=80,
            alpha=0.05,
            seed=2026,
        )
        suffix = "_quick"
    else:
        config = SimConfig(
            sample_sizes=(20, 40, 80, 160),
            estimators=("argmin", "median", "trimmed"),
            B=500,
            R=300,
            alpha=0.05,
            seed=2026,
        )
        suffix = ""

    specs = default_h0_specs() + default_ha_specs()
    print(f"Distribuciones: {[s.name for s in specs]}")
    print(f"Tamaños n: {config.sample_sizes}")
    print(f"B={config.B}, R={config.R}, alpha={config.alpha}")

    df = run_simulation_parallel(specs, config, n_workers=n_workers)
    summary = summarize(df, alpha=config.alpha)

    raw_csv = data_dir / f"tn_simulation_raw{suffix}.csv"
    sum_csv = data_dir / f"tn_simulation_summary{suffix}.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)
    print(f"\nResultados guardados:")
    print(f"  {raw_csv}")
    print(f"  {sum_csv}")

    print("\nResumen (tasas de rechazo):")
    print(
        summary.pivot_table(
            index=["dist", "estimator"], columns="n", values="reject_rate"
        ).round(3).to_string()
    )

    print("\nGenerando gráficas...")
    for fn, name in [
        (plot_type_i_error, "Error Tipo I"),
        (plot_power_curves, "Curvas de potencia"),
        (plot_runtime, "Tiempo de ejecución"),
        (plot_power_vs_cost, "Potencia vs costo"),
        (plot_pvalue_distribution_h0, "Distribución p-valor H0"),
    ]:
        try:
            if fn is plot_type_i_error:
                out = fn(summary, alpha=config.alpha, outdir=fig_dir)
            elif fn is plot_pvalue_distribution_h0:
                out = fn(df, outdir=fig_dir)
            else:
                out = fn(summary, outdir=fig_dir)
            if out:
                print(f"  {name}: {out.name}")
        except Exception as e:
            print(f"  [WARN] {name}: {type(e).__name__}: {e}")

    print("\nListo.")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    main(quick=quick)
