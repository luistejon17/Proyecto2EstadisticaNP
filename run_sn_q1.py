"""
Script temporal: corre solo S_n con q=1 (median + trimmed, B=199, R=200)
y actualiza results/data/sn_simulation_*.csv manteniendo los datos q=2.
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def main():
    import pandas as pd
    from src.distributions import (
        uniform_h0, cauchy_h0, gamma_ha, weibull_ha, pareto_ha,
    )
    from src.simulation_sn import SimConfigSn, summarize_sn
    from src.simulation_sn_parallel import run_simulation_sn_parallel

    RESULTS = ROOT / "results" / "data"
    N_WORKERS = max(1, os.cpu_count() or 2)
    ALL_SPECS = [
        uniform_h0(1.0, 3.0), cauchy_h0(2.0, 1.0),
        gamma_ha(2.0, 1.0), weibull_ha(1.5, 1.0), pareto_ha(3.0, 1.0),
    ]
    WEIGHTS = ("gauss_1.0", "gauss_0.5", "laplace_1.0")

    config_q1 = SimConfigSn(
        sample_sizes=(20, 40, 80, 160),
        estimators=("median", "trimmed"),
        qs=(1,),
        weight_names=WEIGHTS,
        B=199, R=200, alpha=0.05, seed=2026, n_t_grid=301,
    )

    total = (len(ALL_SPECS) * len(config_q1.sample_sizes)
             * len(config_q1.estimators) * len(config_q1.weight_names) * config_q1.R)
    print(f"Tareas q=1: {total}  |  Workers: {N_WORKERS}  |  B=199, R=200")
    print("=" * 60)

    t0 = time.perf_counter()
    df_q1 = run_simulation_sn_parallel(ALL_SPECS, config_q1, n_workers=N_WORKERS)
    elapsed = time.perf_counter() - t0
    print(f"\nq=1 completado en {elapsed/60:.1f} min")

    # Combinar con resultados q=2 existentes (si los hay)
    raw_path = RESULTS / "sn_simulation_raw.csv"
    if raw_path.exists():
        df_existing = pd.read_csv(raw_path)
        df_q2 = df_existing[df_existing["q"] == 2]
        df = pd.concat([df_q2, df_q1], ignore_index=True)
        print(f"Combinado: {len(df_q2)} filas q=2 + {len(df_q1)} filas q=1")
    else:
        df = df_q1
        print(f"Solo q=1: {len(df_q1)} filas")

    summary = summarize_sn(df, alpha=config_q1.alpha)
    df.to_csv(raw_path, index=False)
    summary.to_csv(RESULTS / "sn_simulation_summary.csv", index=False)
    print(f"Guardado en {RESULTS}")


if __name__ == "__main__":
    main()
