"""
Script de línea de comandos para regenerar los resultados completos.
Equivale a correr los notebooks 01 y 02 con quick=False.

Uso:
    python run_full_simulations.py            # corre ambos tests
    python run_full_simulations.py --tn       # solo T_n
    python run_full_simulations.py --sn       # solo S_n
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Asegura que src/ esté en el path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.distributions import (
    uniform_h0, cauchy_h0, gamma_ha, weibull_ha, pareto_ha,
)
from src.simulation import SimConfig, summarize
from src.simulation_parallel import run_simulation_parallel
from src.simulation_sn import SimConfigSn, summarize_sn
from src.simulation_sn_parallel import run_simulation_sn_parallel

RESULTS = ROOT / "results" / "data"
RESULTS.mkdir(parents=True, exist_ok=True)

N_WORKERS = max(1, os.cpu_count() or 2)

H0_SPECS = [uniform_h0(1.0, 3.0), cauchy_h0(2.0, 1.0)]
HA_SPECS  = [gamma_ha(2.0, 1.0), weibull_ha(1.5, 1.0), pareto_ha(3.0, 1.0)]
ALL_SPECS = H0_SPECS + HA_SPECS


def run_tn():
    print("=" * 60)
    print("TEST 1: T_n  (B=199, R=300, n=20/40/80/160, 11 workers)")
    print("  B=199: alpha*(B+1)=0.05*200=10 in Z (Hall & Wilson 1991)")
    print("=" * 60)
    config = SimConfig(
        sample_sizes=(20, 40, 80, 160),
        estimators=("argmin", "median", "trimmed"),
        B=199, R=300, alpha=0.05, seed=2026,
    )
    total = len(ALL_SPECS) * len(config.sample_sizes) * len(config.estimators) * config.R
    print(f"Total tareas: {total}  |  Workers: {N_WORKERS}")

    t0 = time.perf_counter()
    df = run_simulation_parallel(ALL_SPECS, config, n_workers=N_WORKERS)
    elapsed = time.perf_counter() - t0
    print(f"\nT_n completado en {elapsed/60:.1f} min")

    summary = summarize(df, alpha=config.alpha)
    df.to_csv(RESULTS / "tn_simulation_raw.csv", index=False)
    summary.to_csv(RESULTS / "tn_simulation_summary.csv", index=False)
    print(f"Guardado en {RESULTS}")
    return df, summary


def run_sn():
    print("=" * 60)
    print("TEST 2: S_n  (B=199, R=200, n=20/40/80/160, 12 workers)")
    print("  q=1 y q=2: argmin + median + trimmed")
    print("  argmin: secante batcheada con gradiente analitico (Sec. 1.3.4)")
    print("  B=199: alpha*(B+1)=0.05*200=10 in Z (Hall & Wilson 1991)")
    print("=" * 60)

    WEIGHTS = ("gauss_1.0", "gauss_0.5", "laplace_1.0")

    config = SimConfigSn(
        sample_sizes=(20, 40, 80, 160),
        estimators=("argmin", "median", "trimmed"),
        qs=(1, 2),
        weight_names=WEIGHTS,
        B=199, R=200, alpha=0.05, seed=2026, n_t_grid=301,
    )

    total = (len(ALL_SPECS) * len(config.sample_sizes)
             * len(config.estimators) * len(config.qs)
             * len(config.weight_names) * config.R)
    print(f"Total tareas: {total}  |  Workers: {N_WORKERS}")

    t0 = time.perf_counter()
    df = run_simulation_sn_parallel(ALL_SPECS, config, n_workers=N_WORKERS)
    elapsed = time.perf_counter() - t0
    print(f"S_n completado en {elapsed/60:.1f} min")
    summary = summarize_sn(df, alpha=config.alpha)
    df.to_csv(RESULTS / "sn_simulation_raw.csv", index=False)
    summary.to_csv(RESULTS / "sn_simulation_summary.csv", index=False)
    print(f"Guardado en {RESULTS}")
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tn", action="store_true", help="Solo Test 1 (T_n)")
    parser.add_argument("--sn", action="store_true", help="Solo Test 2 (S_n)")
    args = parser.parse_args()

    run_both = not args.tn and not args.sn

    t_global = time.perf_counter()

    if args.tn or run_both:
        run_tn()

    if args.sn or run_both:
        run_sn()

    total_min = (time.perf_counter() - t_global) / 60
    print(f"\n{'='*60}")
    print(f"TODO completado en {total_min:.1f} min")
