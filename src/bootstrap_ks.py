"""
Procedimiento bootstrap simétrico para el estadístico T_n (Schuster-Barker).

Algoritmo
---------
1. A partir de la muestra original X_1, ..., X_n se estima ``theta_tilde`` con
   el estimador del centro de simetría elegido (argmin / mediana / afeitada).
2. Se calcula el estadístico observado T_n(theta_tilde).
3. Para b = 1, ..., B:
      - Se generan n observaciones Y_1, ..., Y_n por muestreo con reemplazo de
        la *muestra simetrizada* {X_i, 2*theta_tilde - X_i} (esto es equivalente
        a muestrear de la cdf simetrizada sF_n(.; theta_tilde)).
      - Se recalcula theta_tilde^* sobre la remuestra (mismo estimador).
      - Se calcula T_n^* = T_n(Y; theta_tilde^*).
4. El p-valor bootstrap es

       p-valor = ( 1 + #{ T_n^* >= T_n_obs } ) / (B + 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .statistics_ks import ESTIMATORS, Tn_statistic, symmetrized_sample, theta_argmin


# ---------------------------------------------------------------------------
# Resultado de un test bootstrap
# ---------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    statistic: float
    p_value: float
    theta_hat: float
    boot_statistics: np.ndarray
    estimator_name: str
    B: int

    @property
    def reject(self) -> bool:
        return self.p_value < 0.05


# ---------------------------------------------------------------------------
# Test bootstrap T_n
# ---------------------------------------------------------------------------
def bootstrap_test_Tn(
    sample: np.ndarray,
    estimator: str | Callable[[np.ndarray], float] = "argmin",
    B: int = 500,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """
    Test bootstrap de simetría basado en T_n.

    Parameters
    ----------
    sample : np.ndarray
        Muestra X_1, ..., X_n.
    estimator : str or callable
        Estimador del centro de simetría. Si es str, se busca en
        ``ESTIMATORS``: ``"argmin"``, ``"median"``, ``"trimmed"``.
    B : int
        Número de remuestras bootstrap.
    rng : np.random.Generator or None
        Generador aleatorio. Si es None se usa ``np.random.default_rng()``.

    Returns
    -------
    BootstrapResult
        Estadístico observado, p-valor bootstrap y demás información.
    """
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(estimator, str):
        est_name = estimator
        est_fn = ESTIMATORS[estimator]
    else:
        est_name = getattr(estimator, "__name__", "custom")
        est_fn = estimator

    x = np.asarray(sample, dtype=float)
    n = x.size

    theta_hat = est_fn(x)
    T_obs = Tn_statistic(x, theta_hat)

    # Soporte simetrizado: 2n puntos con masas iguales (= sF_n(.; theta_hat)).
    support = symmetrized_sample(x, theta_hat)

    # Estimador interior del bootstrap: para argmin usamos un Walsh localizado
    # anclado en theta_hat con k = max(16, n) candidatos (en vez de k = 8n).
    # Justificación: las remuestras se extraen de sF_n(theta_hat), que es
    # simétrica *exactamente* en theta_hat, por lo que el argmin de cada
    # remuestra está a distancia O(1/sqrt(n)) de theta_hat. Anclar en theta_hat
    # (en lugar de en la mediana de la remuestra) garantiza que los k Walsh
    # averages cubren la zona correcta incluso con k pequeño; empíricamente
    # k = n alcanza una tasa de acierto del mínimo global comparable a k = 8n.
    if estimator == "argmin":
        _theta_hat = theta_hat          # captura en el closure
        _k_inner = max(16, n)           # 8x menos candidatos que k = 8n
        def inner_est_fn(y: np.ndarray) -> float:
            return theta_argmin(y, n_walsh=_k_inner, anchor=_theta_hat)
    else:
        inner_est_fn = est_fn

    T_boot = np.empty(B, dtype=float)
    for b in range(B):
        y = rng.choice(support, size=n, replace=True)
        theta_b = inner_est_fn(y)
        T_boot[b] = Tn_statistic(y, theta_b)

    # p-valor bootstrap (con corrección de continuidad +1 / B+1)
    p_val = (1.0 + np.sum(T_boot >= T_obs)) / (B + 1.0)

    return BootstrapResult(
        statistic=T_obs,
        p_value=float(p_val),
        theta_hat=theta_hat,
        boot_statistics=T_boot,
        estimator_name=est_name,
        B=B,
    )
