"""
Script: estudio Monte Carlo paralelizado del Test 2 (S_n).

Ejecución:
    python scripts/run_test2_simulation_parallel.py            # full
    python scripts/run_test2_simulation_parallel.py --quick    # rápido

IMPORTANTE (Windows): el bloque ``if __name__ == '__main__':`` es
obligatorio para que el spawn de los workers funcione.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(quick: bool = False) -> None:
    import os
    import pandas as pd

    from src.distributions import default_h0_specs, default_ha_specs
    from src.plotting_sn import (
        plot_power_curves_sn,
        plot_power_heatmap,
        plot_pvalue_distribution_h0_sn,
        plot_q1_vs_q2,
        plot_runtime_sn,
        plot_tn_vs_sn_power,
        plot_tn_vs_sn_runtime,
        plot_type_i_error_sn,
        plot_weights_compare,
    )
    from src.simulation_sn import SimConfigSn, summarize_sn
    from src.simulation_sn_parallel import run_simulation_sn_parallel

    data_dir = ROOT / "results" / "data"
    fig_dir = ROOT / "results" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_workers = max(1, os.cpu_count() or 2)
    print(f"CPU cores: {os.cpu_count()}  |  Workers: {n_workers}")

    if quick:
        config = SimConfigSn(
            sample_sizes=(20, 40, 80),
            estimators=("argmin", "median", "trimmed"),
            qs=(1, 2),
            weight_names=("gauss_1.0", "laplace_1.0"),
            B=199,
            R=60,
            alpha=0.05,
            seed=2026,
            n_t_grid=201,
        )
        suffix = "_quick"
    else:
        config = SimConfigSn(
            sample_sizes=(20, 40, 80, 160),
            estimators=("argmin", "median", "trimmed"),
            qs=(1, 2),
            weight_names=("gauss_1.0", "gauss_0.5", "laplace_1.0"),
            B=500,
            R=200,
            alpha=0.05,
            seed=2026,
            n_t_grid=301,
        )
        suffix = ""

    specs = default_h0_specs() + default_ha_specs()
    print(f"Distribuciones: {[s.name for s in specs]}")
    print(f"Tamaños n: {config.sample_sizes}")
    print(f"Estimadores: {config.estimators}")
    print(f"q: {config.qs}, pesos: {config.weight_names}")
    print(f"B={config.B}, R={config.R}, K (grid t)={config.n_t_grid}")

    df = run_simulation_sn_parallel(specs, config, n_workers=n_workers)
    summary = summarize_sn(df, alpha=config.alpha)

    raw_csv = data_dir / f"sn_simulation_raw{suffix}.csv"
    sum_csv = data_dir / f"sn_simulation_summary{suffix}.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(sum_csv, index=False)
    print(f"\nResultados guardados:\n  {raw_csv}\n  {sum_csv}")

    # Resumen consola
    print("\nTasas de rechazo (argmin, q=2, pesos en columnas):")
    sub = summary[(summary["estimator"] == "argmin") & (summary["q"] == 2)]
    if not sub.empty:
        pv = sub.pivot_table(
            index=["dist", "n"], columns="weight", values="reject_rate"
        ).round(3)
        print(pv.to_string())

    # Gráficas
    print("\nGenerando gráficas...")
    type_i_paths = plot_type_i_error_sn(summary, alpha=config.alpha, outdir=fig_dir)
    print(f"  Error Tipo I: {len(type_i_paths)} figuras")
    power_paths = plot_power_curves_sn(summary, outdir=fig_dir)
    print(f"  Potencia: {len(power_paths)} figuras")

    for fn, name in [
        (plot_q1_vs_q2, "Comparación q=1 vs q=2"),
        (plot_weights_compare, "Comparación de pesos"),
        (plot_power_heatmap, "Heatmap de potencia"),
        (plot_runtime_sn, "Tiempos"),
    ]:
        try:
            out = fn(summary, outdir=fig_dir)
            if out:
                print(f"  {name}: {out.name}")
        except Exception as e:
            print(f"  [WARN] {name}: {type(e).__name__}: {e}")

    try:
        out = plot_pvalue_distribution_h0_sn(df, outdir=fig_dir)
        if out:
            print(f"  p-valor bajo H0: {out.name}")
    except Exception as e:
        print(f"  [WARN] p-valor H0: {type(e).__name__}: {e}")

    # Comparativas con T_n si hay datos
    tn_csv = data_dir / "tn_simulation_summary.csv"
    if tn_csv.exists():
        print("\nComparativas T_n vs S_n...")
        tn_sum = pd.read_csv(tn_csv)
        try:
            out = plot_tn_vs_sn_power(tn_sum, summary, outdir=fig_dir)
            if out:
                print(f"  Potencia T_n vs S_n: {out.name}")
        except Exception as e:
            print(f"  [WARN] Potencia comparativa: {type(e).__name__}: {e}")
        try:
            out = plot_tn_vs_sn_runtime(tn_sum, summary, outdir=fig_dir)
            if out:
                print(f"  Runtime T_n vs S_n: {out.name}")
        except Exception as e:
            print(f"  [WARN] Runtime comparativo: {type(e).__name__}: {e}")
    else:
        print("\nNo se encontró tn_simulation_summary.csv; saltando comparativas.")

    print("\nListo.")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    main(quick=quick)
