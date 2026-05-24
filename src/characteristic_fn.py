"""
Estadístico S_n basado en la función característica empírica
(Csörgő & Heathcote, 1987; Feuerverger & Mureika, 1977).

Definición
----------
Para una muestra X_1,...,X_n con función característica empírica

    c_n(t) = (1/n) Σ_j exp(i t X_j),

y un estimador del centro de simetría θ̂, el estadístico es

    S_n(θ̂) = ∫ | c_n(t) e^{-it θ̂} - c_s(t) |^q  w(t) dt,

con c_s(t) = |c_n(t)| (la "simetrización"), q ∈ {1, 2} y w una densidad
simétrica.

Computación
-----------
Descomponiendo c_n(t) e^{-it θ̂} en parte real A_n(t;θ) e imaginaria
B_n(t;θ), un cálculo directo da

    | c_n(t) e^{-it θ̂} - c_s(t) |^2  =  2 |c_n(t)| ( |c_n(t)| - A_n(t;θ̂) ).

Además A_n(t;θ) = a_n(t) cos(tθ) + b_n(t) sin(tθ), con
a_n(t) = (1/n) Σ cos(t X_j), b_n(t) = (1/n) Σ sin(t X_j); y |c_n(t)| no
depende de θ. Esto permite precomputar (a_n, b_n) una sola vez por
muestra y barrer múltiples θ con un costo lineal en θ.

La integral se aproxima por regla trapezoidal sobre un grid uniforme
[-t_max, t_max] del que la masa de w(·) fuera del intervalo es
despreciable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import trim_mean


# ---------------------------------------------------------------------------
# Funciones de peso w(t) (simétricas alrededor de 0)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightFn:
    """Función de peso simétrica para la integral S_n.

    Attributes
    ----------
    name : str
        Identificador corto.
    fn : callable
        w(t) sobre arrays numpy.
    t_max : float
        Cota práctica del soporte (fuera de [-t_max, t_max] la masa es ~0).
    """
    name: str
    fn: Callable[[np.ndarray], np.ndarray]
    t_max: float


def gaussian_weight(sigma: float = 1.0) -> WeightFn:
    """Densidad gaussiana N(0, sigma^2)."""
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))

    def fn(t: np.ndarray) -> np.ndarray:
        return norm * np.exp(-(t ** 2) / (2.0 * sigma ** 2))

    return WeightFn(name=f"gauss(s={sigma})", fn=fn, t_max=6.0 * sigma)


def laplace_weight(scale: float = 1.0) -> WeightFn:
    """Densidad Laplace (exponencial doble) f(t) = (a/2) exp(-a|t|)."""
    def fn(t: np.ndarray) -> np.ndarray:
        return 0.5 * scale * np.exp(-scale * np.abs(t))

    return WeightFn(name=f"laplace(a={scale})", fn=fn, t_max=10.0 / scale)


def cauchy_weight(gamma: float = 1.0) -> WeightFn:
    """Densidad Cauchy f(t) = γ / (π (γ^2 + t^2))."""
    def fn(t: np.ndarray) -> np.ndarray:
        return gamma / (np.pi * (gamma ** 2 + t ** 2))

    return WeightFn(name=f"cauchy(g={gamma})", fn=fn, t_max=40.0 * gamma)


def uniform_weight(T: float = 5.0) -> WeightFn:
    """Indicador uniforme en [-T, T] con masa 1."""
    inv2T = 0.5 / T

    def fn(t: np.ndarray) -> np.ndarray:
        return np.where(np.abs(t) <= T, inv2T, 0.0)

    return WeightFn(name=f"unif(T={T})", fn=fn, t_max=T)


def default_weights() -> dict[str, WeightFn]:
    """Catálogo por defecto."""
    return {
        "gauss_1.0": gaussian_weight(1.0),
        "gauss_0.5": gaussian_weight(0.5),
        "laplace_1.0": laplace_weight(1.0),
    }


# ---------------------------------------------------------------------------
# Grid de integración
# ---------------------------------------------------------------------------
def make_t_grid(w_fn: WeightFn, n_points: int = 301) -> np.ndarray:
    """Grid uniforme en [-t_max, t_max] con `n_points` puntos."""
    return np.linspace(-w_fn.t_max, w_fn.t_max, n_points)


# ---------------------------------------------------------------------------
# Estadístico S_n
# ---------------------------------------------------------------------------
def Sn_statistic(
    sample: np.ndarray,
    theta: float,
    w_fn: WeightFn,
    q: int = 2,
    t_grid: np.ndarray | None = None,
) -> float:
    """Calcula S_n(θ) por cuadratura trapezoidal.

    Parameters
    ----------
    sample : np.ndarray
        Muestra X_1,...,X_n.
    theta : float
        Estimador del centro de simetría.
    w_fn : WeightFn
        Función de peso simétrica.
    q : int
        Exponente (1 o 2).
    t_grid : np.ndarray, optional
        Grid de evaluación; si None usa `make_t_grid(w_fn)`.

    Returns
    -------
    float
        S_n(θ).
    """
    if t_grid is None:
        t_grid = make_t_grid(w_fn)
    return float(Sn_multi_theta(sample, np.array([theta]), w_fn, q, t_grid)[0])


def Sn_multi_theta(
    sample: np.ndarray,
    thetas: np.ndarray,
    w_fn: WeightFn,
    q: int = 2,
    t_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorización de S_n(θ) sobre múltiples θ.

    Aprovecha la descomposición A_n(t;θ) = a_n(t) cos(tθ) + b_n(t) sin(tθ)
    y que |c_n(t)| no depende de θ: precomputa a_n, b_n, |c_n| una vez y
    luego barre los M valores de θ con costo O(M K).
    """
    if t_grid is None:
        t_grid = make_t_grid(w_fn)

    x = np.asarray(sample, dtype=float)
    th = np.atleast_1d(np.asarray(thetas, dtype=float))
    K = t_grid.size

    # c_n(t) sin centrar: a_n(t) + i b_n(t)
    arg_raw = t_grid[:, None] * x[None, :]              # (K, n)
    a_n = np.mean(np.cos(arg_raw), axis=1)              # (K,)
    b_n = np.mean(np.sin(arg_raw), axis=1)              # (K,)
    abs_cn = np.sqrt(a_n ** 2 + b_n ** 2)               # (K,)

    # Para cada θ_m: A_n(t;θ_m) = a_n(t) cos(tθ_m) + b_n(t) sin(tθ_m)
    tth = t_grid[:, None] * th[None, :]                 # (K, M)
    A = a_n[:, None] * np.cos(tth) + b_n[:, None] * np.sin(tth)  # (K, M)

    abs_cn_col = abs_cn[:, None]                        # (K, 1)
    # |c_n e^{-itθ} - c_s|^2 = 2 |c_n| (|c_n| - A)
    diff_sq = 2.0 * abs_cn_col * (abs_cn_col - A)
    np.maximum(diff_sq, 0.0, out=diff_sq)               # protección numérica

    if q == 2:
        integrand = diff_sq
    elif q == 1:
        integrand = np.sqrt(diff_sq)
    else:
        integrand = diff_sq ** (q / 2.0)

    weights = w_fn.fn(t_grid)[:, None]                  # (K, 1)
    # trapezoidal: ∫ integrand(t) w(t) dt
    return np.trapezoid(integrand * weights, t_grid, axis=0)  # (M,)


# ---------------------------------------------------------------------------
# Estimadores del centro de simetría
# ---------------------------------------------------------------------------
def theta_median(sample: np.ndarray) -> float:
    """Mediana muestral."""
    return float(np.median(sample))


def theta_trimmed(sample: np.ndarray, trim: float = 0.1) -> float:
    """Media afeitada (trimmed mean)."""
    return float(trim_mean(sample, proportiontocut=trim))


def theta_argmin_Sn(
    sample: np.ndarray,
    w_fn: WeightFn,
    q: int = 2,
    bracket: tuple[float, float] | None = None,
    n_grid: int = 40,
    t_grid: np.ndarray | None = None,
) -> float:
    """argmin de S_n(θ) sobre θ.

    Búsqueda en dos etapas:
      1. Grid uniforme en el rango muestral ampliado un 10%.
      2. Refinamiento Brent local alrededor del mejor punto del grid.
    """
    x = np.asarray(sample, dtype=float)
    if bracket is None:
        lo, hi = float(x.min()), float(x.max())
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        bracket = (lo - pad, hi + pad)
    lo, hi = bracket

    if t_grid is None:
        t_grid = make_t_grid(w_fn)

    grid = np.linspace(lo, hi, n_grid)
    vals = Sn_multi_theta(x, grid, w_fn=w_fn, q=q, t_grid=t_grid)
    k = int(np.argmin(vals))
    th0 = float(grid[k])

    left = float(grid[max(k - 1, 0)])
    right = float(grid[min(k + 1, n_grid - 1)])
    if left >= right:
        return th0

    try:
        res = minimize_scalar(
            lambda th: Sn_statistic(x, th, w_fn, q, t_grid),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-5},
        )
        return float(res.x)
    except Exception:
        return th0


# ---------------------------------------------------------------------------
# Catálogo de estimadores indexados por nombre
# ---------------------------------------------------------------------------
def get_estimator(
    name: str,
    w_fn: WeightFn,
    q: int,
    t_grid: np.ndarray | None = None,
) -> Callable[[np.ndarray], float]:
    """Devuelve la función estimadora dado el nombre y la config (w, q)."""
    if name == "argmin":
        return lambda x: theta_argmin_Sn(x, w_fn=w_fn, q=q, t_grid=t_grid)
    if name == "median":
        return theta_median
    if name == "trimmed":
        return theta_trimmed
    raise ValueError(f"Estimador '{name}' no reconocido.")
