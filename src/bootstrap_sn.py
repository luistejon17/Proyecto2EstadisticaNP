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


def bootstrap_test_Sn(
    sample: np.ndarray,
    w_fn: WeightFn,
    q: int = 2,
    estimator: str | Callable[[np.ndarray], float] = "argmin",
    B: int = 500,
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
        Remuestras bootstrap.
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

    # Soporte simetrizado para muestreo bootstrap
    support = np.concatenate([x, 2.0 * theta_hat - x])

    S_boot = np.empty(B, dtype=float)
    for b in range(B):
        y = rng.choice(support, size=n, replace=True)
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
