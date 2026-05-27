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
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import trim_mean

from .pyomo_argmin import solve_discrete_argmin_pyomo


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


def _Sn_precompute(
    sample: np.ndarray, w_fn: WeightFn, t_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precomputa (a_n(t), b_n(t), |c_n|(t), w(t)) — independientes de θ."""
    x = np.asarray(sample, dtype=float)
    arg = t_grid[:, None] * x[None, :]
    a_n = np.mean(np.cos(arg), axis=1)
    b_n = np.mean(np.sin(arg), axis=1)
    abs_cn = np.sqrt(a_n ** 2 + b_n ** 2)
    w_vals = w_fn.fn(t_grid)
    return a_n, b_n, abs_cn, w_vals


def _Sn_value_and_grad_q1(
    theta: float,
    a_n: np.ndarray,
    b_n: np.ndarray,
    abs_cn: np.ndarray,
    w_vals: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[float, float]:
    """
    Evalúa S_n(θ) y su gradiente analítico ∂S_n/∂θ para q=1.

    Usando la identidad 1 - cos(u) = 2 sin²(u/2):
        |c_n e^{-itθ} - |c_n|| = 2 |c_n| |sin((Φ_n - tθ)/2)|

    El gradiente existe para casi todo θ (los ceros del seno forman un
    conjunto de medida cero en t para cada θ fijo):
        ∂S_n/∂θ = -∫ t |c_n| sgn(sin((Φ_n-tθ)/2)) cos((Φ_n-tθ)/2) w dt
    """
    Phi_n = np.arctan2(b_n, a_n)
    half = (Phi_n - t_grid * theta) / 2.0
    sin_h = np.sin(half)
    cos_h = np.cos(half)

    integrand = 2.0 * abs_cn * np.abs(sin_h)
    S_n = float(np.trapezoid(integrand * w_vals, t_grid))

    grad_integ = -t_grid * abs_cn * np.sign(sin_h) * cos_h * w_vals
    dS_n = float(np.trapezoid(grad_integ, t_grid))
    return S_n, dS_n


def _Sn_value_and_grad_q2(
    theta: float,
    a_n: np.ndarray,
    b_n: np.ndarray,
    abs_cn: np.ndarray,
    w_vals: np.ndarray,
    t_grid: np.ndarray,
) -> tuple[float, float]:
    """
    Evalúa S_n(θ) y su gradiente analítico ∂S_n/∂θ para q=2.

    Usando la descomposición c_n(t) e^{-itθ} = A_n + i B_n con
        A_n = a_n cos(tθ) + b_n sin(tθ)
        B_n = -a_n sin(tθ) + b_n cos(tθ)
    y la identidad |c_n e^{-itθ} - |c_n||² = 2|c_n|(|c_n| - A_n),

        S_n(θ)  = ∫ 2 |c_n| (|c_n| - A_n) w(t) dt
        ∂S_n/∂θ = -2 ∫ t |c_n| B_n w(t) dt

    (equivalente a la forma clásica -2 ∫ t |c_n|² sin(Φ_n - tθ) w dt).
    """
    t_theta = t_grid * theta
    cos_tt = np.cos(t_theta)
    sin_tt = np.sin(t_theta)
    A = a_n * cos_tt + b_n * sin_tt
    B = -a_n * sin_tt + b_n * cos_tt

    integrand = 2.0 * abs_cn * (abs_cn - A)
    np.maximum(integrand, 0.0, out=integrand)
    S_n = float(np.trapezoid(integrand * w_vals, t_grid))

    grad_integ = -2.0 * t_grid * abs_cn * B * w_vals
    dS_n = float(np.trapezoid(grad_integ, t_grid))
    return S_n, dS_n


def theta_argmin_Sn(
    sample: np.ndarray,
    w_fn: WeightFn,
    q: int = 2,
    bracket: tuple[float, float] | None = None,
    t_grid: np.ndarray | None = None,
    theta0: float | None = None,
) -> float:
    """argmin de S_n(θ) sobre θ.

    Para ``q=2`` usa L-BFGS-B con gradiente analítico (Leibniz):
        ∂S_n/∂θ = -2 ∫ t |c_n|² sin(Φ_n(t) - tθ) w(t) dt.
    El precompute de (a_n, b_n, |c_n|, w) se hace una vez (O(K n)) y cada
    iteración del optimizador cuesta O(K). El punto de partida por defecto
    es la mediana muestral.

    Para ``q != 2`` (donde el integrando contiene una raíz no diferenciable
    en ceros) se recurre a la implementación de grid + Brent.
    """
    x = np.asarray(sample, dtype=float)
    if t_grid is None:
        t_grid = make_t_grid(w_fn)

    if bracket is None:
        lo, hi = float(x.min()), float(x.max())
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        bracket = (lo - pad, hi + pad)

    if q not in (1, 2):
        return _theta_argmin_Sn_grid(
            x, w_fn, q=q, bracket=bracket, t_grid=t_grid,
        )

    a_n, b_n, abs_cn, w_vals = _Sn_precompute(x, w_fn, t_grid)

    if theta0 is None:
        theta0 = float(np.median(x))

    grad_fn = _Sn_value_and_grad_q1 if q == 1 else _Sn_value_and_grad_q2

    def fun(theta_arr):
        S, dS = grad_fn(float(theta_arr[0]), a_n, b_n, abs_cn, w_vals, t_grid)
        return S, np.array([dS])

    res = minimize(
        fun, x0=np.array([theta0]), jac=True,
        method="L-BFGS-B", bounds=[bracket],
        options={"gtol": 1e-7, "ftol": 1e-10, "maxiter": 50},
    )
    return float(res.x[0])


def _theta_argmin_Sn_grid(
    x: np.ndarray,
    w_fn: WeightFn,
    q: int,
    bracket: tuple[float, float],
    t_grid: np.ndarray,
    n_grid: int = 40,
) -> float:
    """Variante grid + Brent (usada como fallback para q != 2)."""
    lo, hi = bracket
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


def theta_argmin_Sn_grid(
    sample: np.ndarray,
    w_fn: WeightFn,
    q: int = 2,
    bracket: tuple[float, float] | None = None,
    n_grid: int = 40,
    t_grid: np.ndarray | None = None,
) -> float:
    """Variante explícita grid + Brent (conservada para comparación)."""
    x = np.asarray(sample, dtype=float)
    if t_grid is None:
        t_grid = make_t_grid(w_fn)
    if bracket is None:
        lo, hi = float(x.min()), float(x.max())
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        bracket = (lo - pad, hi + pad)
    return _theta_argmin_Sn_grid(x, w_fn, q, bracket, t_grid, n_grid)


def theta_argmin_Sn_pyomo(
    sample: np.ndarray,
    w_fn: WeightFn,
    q: int = 2,
    bracket: tuple[float, float] | None = None,
    n_theta_grid: int = 40,
    t_grid: np.ndarray | None = None,
    *,
    solver: str | None = None,
    solver_options: dict | None = None,
    tee: bool = False,
) -> float:
    """Argmin de S_n seleccionado con Pyomo sobre una grilla de theta.

    Esta variante construye una grilla uniforme en el intervalo de búsqueda,
    evalúa ``S_n`` en cada punto y usa Pyomo para seleccionar el candidato con
    menor valor. Es útil para correr las pruebas con un estimador explícitamente
    seleccionado por Pyomo sin cambiar el resto del pipeline.
    """
    x = np.asarray(sample, dtype=float)
    if x.size == 0:
        return 0.0
    if t_grid is None:
        t_grid = make_t_grid(w_fn)
    if bracket is None:
        lo, hi = float(x.min()), float(x.max())
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        bracket = (lo - pad, hi + pad)

    if n_theta_grid < 2:
        raise ValueError("n_theta_grid debe ser al menos 2.")

    lo, hi = bracket
    cands = np.linspace(lo, hi, n_theta_grid)
    vals = Sn_multi_theta(x, cands, w_fn=w_fn, q=q, t_grid=t_grid)
    return solve_discrete_argmin_pyomo(
        cands,
        vals,
        solver=solver,
        solver_options=solver_options,
        tee=tee,
    )


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
    if name in ("argmin_pyomo", "argmin_pyomo_grid"):
        return lambda x: theta_argmin_Sn_pyomo(x, w_fn=w_fn, q=q, t_grid=t_grid)
    if name == "median":
        return theta_median
    if name == "trimmed":
        return theta_trimmed
    raise ValueError(f"Estimador '{name}' no reconocido.")
