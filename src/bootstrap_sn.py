"""
Procedimiento bootstrap para S_n (análogo al de T_n).

Algoritmo
---------
1. Estimar θ̂ con el estimador elegido (argmin de S_n, mediana o afeitada).
2. Calcular S_obs = S_n(X; θ̂, q, w).
3. Construir el soporte simetrizado {X_i, 2θ̂ - X_i} (2n puntos con masa
   uniforme = la cdf simetrizada sF_n(·; θ̂)).
4. Para b = 1,...,B:
     - Remuestrear Y_1,...,Y_n con reemplazo del soporte simetrizado.
     - Recalcular θ̂* sobre la remuestra.
     - Calcular S_n^*(b) = S_n(Y; θ̂*, q, w).
5. p-valor = (1 + #{S_n^*(b) >= S_obs}) / (B+1).

Optimizaciones
--------------
- Para estimadores median y trimmed, todas las B remuestras se generan de
  golpe como array (B, n) y los estadísticos S_n^*(b) se calculan en lote
  con `_sn_boot_batch`, que vectoriza sobre b con arreglos (B, K, n).
  El cálculo se hace en mini-lotes para no exceder ~200 MB de RAM.
- Para argmin, la optimización iterativa (L-BFGS-B) impide vectorizar sobre b;
  se mantiene el bucle secuencial.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .characteristic_fn import (
    Sn_statistic,
    WeightFn,
    get_estimator,
    make_t_grid,
)


@dataclass
class BootstrapResultSn:
    statistic: float
    p_value: float
    theta_hat: float
    boot_statistics: np.ndarray
    estimator_name: str
    weight_name: str
    q: int
    B: int

    @property
    def reject(self) -> bool:
        return self.p_value < 0.05


# ---------------------------------------------------------------------------
# Cálculo vectorizado de S_n para B remuestras (median / trimmed)
# ---------------------------------------------------------------------------
def _sn_boot_batch(
    Y: np.ndarray,
    thetas: np.ndarray,
    w_fn: WeightFn,
    q: int,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Evalúa S_n sobre B remuestras simultáneamente.

    Parameters
    ----------
    Y : (B, n) array  — remuestras bootstrap
    thetas : (B,) array — estimadores del centro para cada remuestra
    w_fn : WeightFn
    q : int (1 o 2)
    t_grid : (K,) array

    Returns
    -------
    S_boot : (B,) array
    """
    B_total, n = Y.shape
    K = t_grid.size
    w_vals = w_fn.fn(t_grid)                            # (K,)

    # Tamaño de lote: ~200 MB ÷ (K × n × 8 bytes por float64)
    B_sub = max(1, int(200_000_000 // (K * n * 8)))

    S_boot = np.empty(B_total, dtype=float)
    for start in range(0, B_total, B_sub):
        end = min(start + B_sub, B_total)
        Yb = Y[start:end]           # (bs, n)
        tb = thetas[start:end]      # (bs,)

        arg = t_grid[None, :, None] * Yb[:, None, :]   # (bs, K, n)
        a_n = np.mean(np.cos(arg), axis=2)             # (bs, K)
        b_n = np.mean(np.sin(arg), axis=2)
        abs_cn = np.sqrt(a_n**2 + b_n**2)

        tth = t_grid[None, :] * tb[:, None]            # (bs, K)
        A = a_n * np.cos(tth) + b_n * np.sin(tth)
        diff_sq = 2.0 * abs_cn * (abs_cn - A)
        np.maximum(diff_sq, 0.0, out=diff_sq)          # estabilidad numérica

        integrand = diff_sq if q == 2 else np.sqrt(diff_sq)
        S_boot[start:end] = np.trapezoid(
            integrand * w_vals[None, :], t_grid, axis=1
        )
    return S_boot


# ---------------------------------------------------------------------------
# Test bootstrap S_n
# ---------------------------------------------------------------------------
def bootstrap_test_Sn(
    sample: np.ndarray,
    w_fn: WeightFn,
    q: int = 2,
    estimator: str | Callable[[np.ndarray], float] = "argmin",
    B: int = 199,
    t_grid: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> BootstrapResultSn:
    """Test bootstrap de simetría basado en S_n.

    Parameters
    ----------
    sample : np.ndarray
        Muestra X_1,...,X_n.
    w_fn : WeightFn
        Función de peso simétrica.
    q : int
        Exponente, 1 o 2.
    estimator : str or callable
        "argmin", "median", "trimmed", o una callable.
    B : int
        Remuestras bootstrap. Usar B tal que α(B+1)∈ℤ; para α=0.05
        las opciones válidas son 19, 39, 99, 199, 499, … (Hall & Wilson 1991).
    t_grid : np.ndarray, optional
        Grid de evaluación de S_n. Si None se construye con `make_t_grid(w_fn)`.
    rng : np.random.Generator, optional

    Returns
    -------
    BootstrapResultSn
    """
    if rng is None:
        rng = np.random.default_rng()

    if t_grid is None:
        t_grid = make_t_grid(w_fn)

    if isinstance(estimator, str):
        est_name = estimator
        est_fn = get_estimator(estimator, w_fn, q, t_grid)
    else:
        est_name = getattr(estimator, "__name__", "custom")
        est_fn = estimator

    x = np.asarray(sample, dtype=float)
    n = x.size

    theta_hat = est_fn(x)
    S_obs = Sn_statistic(x, theta_hat, w_fn, q, t_grid)

    # Soporte simetrizado: 2n puntos con masas iguales
    support = np.concatenate([x, 2.0 * theta_hat - x])

    # Generar todas las B remuestras de golpe
    Y_boot = rng.choice(support, size=(B, n), replace=True)

    if est_name in ("median", "trimmed"):
        # --- Camino vectorizado: thetas en batch, S_n en lote ---
        if est_name == "median":
            thetas_b = np.median(Y_boot, axis=1)           # (B,)
        else:
            from scipy.stats import trim_mean
            thetas_b = trim_mean(Y_boot, 0.1, axis=1)      # (B,)
        S_boot = _sn_boot_batch(Y_boot, thetas_b, w_fn, q, t_grid)
    else:
        # --- Camino secuencial: argmin requiere optimización por remuestra ---
        S_boot = np.empty(B, dtype=float)
        for b in range(B):
            y = Y_boot[b]
            theta_b = est_fn(y)
            S_boot[b] = Sn_statistic(y, theta_b, w_fn, q, t_grid)

    p_val = (1.0 + np.sum(S_boot >= S_obs)) / (B + 1.0)

    return BootstrapResultSn(
        statistic=S_obs,
        p_value=float(p_val),
        theta_hat=theta_hat,
        boot_statistics=S_boot,
        estimator_name=est_name,
        weight_name=w_fn.name,
        q=q,
        B=B,
    )
