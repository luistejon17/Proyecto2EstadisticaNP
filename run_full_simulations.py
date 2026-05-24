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

N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

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
    print("TEST 2: S_n  (B=199, R=200, n=20/40/80/160, 11 workers)")
    print("  q=2: argmin + median + trimmed  (L-BFGS-B con gradiente)")
    print("  q=1: median + trimmed           (argmin excluido: q=1 no")
    print("       tiene gradiente analitico, la busqueda en grid es")
    print("       imprecisa y el costo es prohibitivo en el bootstrap)")
    print("  B=199: alpha*(B+1)=0.05*200=10 in Z (Hall & Wilson 1991)")
    print("=" * 60)

    WEIGHTS = ("gauss_1.0", "gauss_0.5", "laplace_1.0")

    # q=2: todos los estimadores (L-BFGS-B rapido y preciso)
    config_q2 = SimConfigSn(
        sample_sizes=(20, 40, 80, 160),
        estimators=("argmin", "median", "trimmed"),
        qs=(2,),
        weight_names=WEIGHTS,
        B=199, R=200, alpha=0.05, seed=2026, n_t_grid=301,
    )
    # q=1: solo median y trimmed (argmin excluido)
    config_q1 = SimConfigSn(
        sample_sizes=(20, 40, 80, 160),
        estimators=("median", "trimmed"),
        qs=(1,),
        weight_names=WEIGHTS,
        B=199, R=200, alpha=0.05, seed=2026, n_t_grid=301,
    )

    total_q2 = (len(ALL_SPECS) * len(config_q2.sample_sizes)
                * len(config_q2.estimators) * len(config_q2.qs)
                * len(config_q2.weight_names) * config_q2.R)
    total_q1 = (len(ALL_SPECS) * len(config_q1.sample_sizes)
                * len(config_q1.estimators) * len(config_q1.qs)
                * len(config_q1.weight_names) * config_q1.R)
    print(f"Tareas q=2: {total_q2}  |  Tareas q=1: {total_q1}  |  Workers: {N_WORKERS}")

    t0 = time.perf_counter()

    print("\n--- Corriendo q=2 ---")
    df_q2 = run_simulation_sn_parallel(ALL_SPECS, config_q2, n_workers=N_WORKERS)
    t_q2 = time.perf_counter() - t0
    print(f"q=2 completado en {t_q2/60:.1f} min")

    print("\n--- Corriendo q=1 ---")
    df_q1 = run_simulation_sn_parallel(ALL_SPECS, config_q1, n_workers=N_WORKERS)
    elapsed = time.perf_counter() - t0
    print(f"q=1 completado. Total S_n: {elapsed/60:.1f} min")

    df = pd.concat([df_q2, df_q1], ignore_index=True)
    summary = summarize_sn(df, alpha=config_q2.alpha)
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
