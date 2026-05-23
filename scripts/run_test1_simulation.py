"""
Script principal: estudio Monte Carlo del Test 1 (T_n, Schuster-Barker).

Ejecución:
    python scripts/run_test1_simulation.py

Salidas:
    results/data/tn_simulation_raw.csv      - una fila por réplica
    results/data/tn_simulation_summary.csv  - resumen por (dist, n, estimador)
    results/figures/tn_*.png                - gráficas
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.distributions import default_h0_specs, default_ha_specs
from src.plotting import (
    plot_power_curves,
    plot_power_vs_cost,
    plot_pvalue_distribution_h0,
    plot_runtime,
    plot_type_i_error,
)
from src.simulation import SimConfig, run_simulation, summarize


def main(quick: bool = False) -> None:
    data_dir = ROOT / "results" / "data"
    fig_dir = ROOT / "results" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        # Configuración rápida para pruebas (~1-2 minutos)
        config = SimConfig(
            sample_sizes=(20, 40, 80),
            estimators=("argmin", "median", "trimmed"),
            B=199,
            R=80,
            alpha=0.05,
            seed=2026,
        )
    else:
        # Configuración del enunciado: B=500, R razonable
        config = SimConfig(
            sample_sizes=(20, 40, 80, 160),
            estimators=("argmin", "median", "trimmed"),
            B=500,
            R=300,
            alpha=0.05,
            seed=2026,
        )

    specs = default_h0_specs() + default_ha_specs()
    print(f"Distribuciones: {[s.name for s in specs]}")
    print(f"Tamaños n: {config.sample_sizes}")
    print(f"Estimadores: {config.estimators}")
    print(f"B={config.B}, R={config.R}")

    df = run_simulation(specs, config)
    summary = summarize(df, alpha=config.alpha)

    raw_csv = data_dir / ("tn_simulation_raw_quick.csv" if quick else "tn_simulation_raw.csv")
    sum_csv = data_dir / ("tn_simulation_summary_quick.csv" if quick else "tn_simulation_summary.csv")
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)
    print(f"\nResultados guardados:")
    print(f"  {raw_csv}")
    print(f"  {sum_csv}")

    print("\nResumen (tasas de rechazo):")
    print(
        summary.pivot_table(
            index=["dist", "estimator"], columns="n", values="reject_rate"
        ).round(3)
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
                print(f"  {name}: {out}")
        except Exception as e:
            print(f"  [WARN] {name}: {type(e).__name__}: {e}")

    print("\nListo.")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    main(quick=quick)
