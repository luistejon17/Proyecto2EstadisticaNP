"""
Estadístico T_n tipo Kolmogorov-Smirnov para el test bootstrap de simetría
(Schuster & Barker, 1987).

Definiciones
------------
Sea X_1, ..., X_n una muestra iid con cdf F. La cdf empírica es

    F_n(x) = (1/n) * sum_i I(X_i <= x).

La *simetrización empírica* respecto a theta es la cdf empírica de los 2n
puntos {X_1, ..., X_n, 2*theta - X_1, ..., 2*theta - X_n}. Equivalentemente

    sF_n(x; theta) = ( F_n(x) + 1 - F_n( (2*theta - x)^- ) ) / 2,

donde F_n(y^-) = (1/n) * sum_i I(X_i < y). Por construcción sF_n(.; theta) es
una cdf simétrica respecto a theta.

El estadístico KS de simetría es

    T_n(theta) = sqrt(n) * sup_x | F_n(x) - sF_n(x; theta) |.

Como F_n(x) - sF_n(x; theta) = ( F_n(x) - 1 + F_n((2*theta - x)^-) ) / 2,
basta evaluar el supremo en los puntos de discontinuidad: la unión de la
muestra original y de su reflexión {2*theta - X_i}.

Estimadores del centro de simetría
----------------------------------
- ``theta_argmin``: argmin de T_n(theta), siguiendo Schuster-Narvarte.
- ``theta_median``: mediana muestral.
- ``theta_trimmed``: media afeitada (trimmed mean) con fracción ``trim``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import trim_mean


# ---------------------------------------------------------------------------
# Estadístico T_n
# ---------------------------------------------------------------------------
def Tn_statistic(sample: np.ndarray, theta: float) -> float:
    """
    Calcula T_n(theta) = sqrt(n) * sup_x |F_n(x) - sF_n(x; theta)|.

    Parameters
    ----------
    sample : np.ndarray
        Muestra de tamaño n (no requiere estar ordenada).
    theta : float
        Valor del centro de simetría.

    Returns
    -------
    float
        Valor del estadístico T_n(theta).
    """
    return float(Tn_multi(sample, np.array([theta], dtype=float))[0])


def Tn_multi(sample: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    """
    Vectorización de :func:`Tn_statistic` sobre múltiples valores de theta.

    Parameters
    ----------
    sample : np.ndarray
        Muestra de tamaño n.
    thetas : np.ndarray
        Vector de K valores de theta.

    Returns
    -------
    np.ndarray
        Vector de tamaño K con los valores T_n(theta_k).
    """
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n == 0:
        return np.zeros_like(thetas, dtype=float)

    x_sorted = np.sort(x)
    th = np.asarray(thetas, dtype=float)

    # Grid de evaluación por theta: union(X_i, 2*theta - X_j). Forma (K, 2n).
    # No es necesario ordenar el grid: searchsorted evalúa cada punto contra
    # x_sorted independientemente.
    refl = 2.0 * th[:, None] - x_sorted[None, :]
    grid = np.concatenate(
        [np.broadcast_to(x_sorted, refl.shape), refl], axis=1
    )

    Fn = np.searchsorted(x_sorted, grid, side="right") / n
    arg = 2.0 * th[:, None] - grid
    Fn_minus = np.searchsorted(x_sorted, arg, side="left") / n

    diff = 0.5 * (Fn - 1.0 + Fn_minus)
    sup = np.max(np.abs(diff), axis=1)
    return np.sqrt(n) * sup


def symmetrized_sample(sample: np.ndarray, theta: float) -> np.ndarray:
    """
    Devuelve la muestra simetrizada {X_i, 2*theta - X_i}.

    Es el soporte (con masas iguales 1/(2n)) de la cdf simetrizada sF_n.
    """
    x = np.asarray(sample, dtype=float)
    return np.concatenate([x, 2.0 * theta - x])


# ---------------------------------------------------------------------------
# Estimadores del centro de simetría
# ---------------------------------------------------------------------------
def theta_median(sample: np.ndarray) -> float:
    """Mediana muestral como estimador del centro de simetría."""
    return float(np.median(sample))


def theta_trimmed(sample: np.ndarray, trim: float = 0.1) -> float:
    """
    Media afeitada (trimmed mean) con fracción ``trim`` en cada cola.

    Parameters
    ----------
    sample : np.ndarray
    trim : float
        Fracción a recortar de cada cola (default 0.1).
    """
    return float(trim_mean(sample, proportiontocut=trim))


def theta_argmin(
    sample: np.ndarray,
    bracket: tuple[float, float] | None = None,
    n_grid: int = 60,
) -> float:
    """
    Argmin del estadístico T_n(theta) sobre theta.

    Implementa una búsqueda en dos etapas:
      1. Evaluar T_n en un grid uniforme dentro del rango de la muestra.
      2. Refinar localmente con búsqueda áurea (Brent) alrededor del mejor punto.

    Parameters
    ----------
    sample : np.ndarray
    bracket : (lo, hi) o None
        Rango sobre el cual buscar theta. Si es None, se usa
        ``(min(x), max(x))`` ampliado un 10%.
    n_grid : int
        Número de puntos en la rejilla inicial.

    Returns
    -------
    float
        Estimación del centro de simetría.
    """
    x = np.asarray(sample, dtype=float)
    if bracket is None:
        lo, hi = float(x.min()), float(x.max())
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        bracket = (lo - pad, hi + pad)
    lo, hi = bracket

    grid = np.linspace(lo, hi, n_grid)
    vals = Tn_multi(x, grid)
    k = int(np.argmin(vals))
    th0 = float(grid[k])

    # Refinamiento Brent en un intervalo local
    left = float(grid[max(k - 1, 0)])
    right = float(grid[min(k + 1, n_grid - 1)])
    if left >= right:
        return th0

    try:
        res = minimize_scalar(
            lambda th: Tn_statistic(x, th),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-6},
        )
        return float(res.x)
    except Exception:
        return th0


# Diccionario de estimadores disponibles -----------------------------------
ESTIMATORS: dict[str, Callable[[np.ndarray], float]] = {
    "argmin": theta_argmin,
    "median": theta_median,
    "trimmed": theta_trimmed,
}
