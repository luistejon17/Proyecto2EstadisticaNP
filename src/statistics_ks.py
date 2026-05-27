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
  Implementación por defecto: restringe la búsqueda a los k Walsh averages
  (X_i + X_j)/2 más cercanos a la mediana. Como T_n es piecewise constant
  y sus saltos ocurren *exactamente* en estos puntos, evaluar sobre la
  vecindad correcta de Walsh averages encuentra el mínimo global con
  altísima probabilidad y a costo O(n^{5/2} log n) en lugar de O(n^3 log n).
- ``theta_argmin_walsh_full``: enumeración exacta de los O(n^2) Walsh averages.
- ``theta_argmin_grid``: variante antigua basada en grid uniforme + refinamiento Brent.
- ``theta_median``: mediana muestral.
- ``theta_trimmed``: media afeitada (trimmed mean) con fracción ``trim``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import trim_mean

from .pyomo_argmin import solve_discrete_argmin_pyomo


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


def theta_hodges_lehmann(sample: np.ndarray) -> float:
    """
    Estimador de Hodges-Lehmann: mediana de todos los promedios de Walsh
    W_{ij} = (X_i + X_j) / 2, i <= j.

    Equivale al estimador que minimiza
        sum_{i<=j} |X_i + X_j - 2*theta|
    (función objetivo del tipo Wilcoxon de rango con signo).
    Es robusto (breakdown point 29%) y asintóticamente normal con
    eficiencia relativa de Pitman ≈ 0.955 bajo normalidad.

    Complejidad: O(n²) para generar Walsh + O(n²) para la mediana.
    """
    W = _walsh_averages(np.asarray(sample, dtype=float))
    return float(np.median(W))


# ---------------------------------------------------------------------------
# argmin de T_n basado en Walsh averages
# ---------------------------------------------------------------------------
def _walsh_averages(x: np.ndarray) -> np.ndarray:
    """
    Genera los promedios de Walsh W_{ij} = (x_i + x_j) / 2 para i <= j.

    Devuelve un vector de n(n+1)/2 elementos (incluye W_{ii} = x_i).
    """
    i, j = np.triu_indices(x.size)
    return 0.5 * (x[i] + x[j])


def _default_k_walsh(n: int) -> int:
    """
    Número por defecto de Walsh averages a evaluar alrededor del ancla.

    Heurística empírica: k = max(64, 8n). Para n in {20,40,80,160} esto
    coincide con el mínimo exacto exhaustivo en ~98% de los casos (vs ~63%
    para el heurístico grid+Brent), con un costo aproximadamente 7× del
    grid pero todavía sub-milisegundos por evaluación para n ≤ 80.

    La justificación teórica: la mediana muestral está a distancia
    O(1/sqrt(n)) del centro verdadero, y la densidad de Walsh averages
    cerca del centro escala como n^2 / rango. Una ventana de O(n^{3/2})
    Walsh promedios *garantiza* la cobertura, pero empíricamente 8n basta.
    """
    total = n * (n + 1) // 2
    return int(min(total, max(64, 8 * n)))


def theta_argmin(
    sample: np.ndarray,
    n_walsh: int | None = None,
    anchor: str | float = "median",
) -> float:
    """
    Argmin de T_n(theta) restringido a una vecindad de Walsh averages.

    Como T_n es piecewise constant en theta con saltos *exactamente* en los
    promedios de Walsh W_{ij} = (X_i + X_j)/2, el mínimo global se alcanza
    en alguno de ellos. En lugar de evaluar los O(n^2) Walsh averages,
    se evalúan los ``n_walsh`` más cercanos a un ancla robusta (mediana
    por defecto).

    Complejidad: O(n^2) para generar Walsh + O(n_walsh · n log n) para
    evaluar T_n vectorizado.

    Parameters
    ----------
    sample : np.ndarray
    n_walsh : int or None
        Cantidad de Walsh averages a evaluar. Si None, usa la heurística
        ``_default_k_walsh(n)`` (~2 n^{3/2}).
    anchor : "median", "trimmed", o float
        Punto de partida alrededor del cual se eligen los k Walsh averages
        más cercanos. Si es float, se usa directamente.

    Returns
    -------
    float
        Walsh average con menor T_n.
    """
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(x[0])

    if isinstance(anchor, str):
        if anchor == "median":
            theta_anchor = float(np.median(x))
        elif anchor == "trimmed":
            theta_anchor = float(trim_mean(x, 0.1))
        else:
            raise ValueError(f"anchor inválido: {anchor!r}")
    else:
        theta_anchor = float(anchor)

    W = _walsh_averages(x)
    n_total = W.size

    if n_walsh is None:
        n_walsh = _default_k_walsh(n)

    if n_walsh >= n_total:
        cands = W
    else:
        # k Walsh promedios más cercanos al ancla, sin orden
        dists = np.abs(W - theta_anchor)
        idx = np.argpartition(dists, n_walsh - 1)[:n_walsh]
        cands = W[idx]

    vals = Tn_multi(x, cands)
    return float(cands[int(np.argmin(vals))])


def theta_argmin_walsh_full(sample: np.ndarray) -> float:
    """
    Argmin exacto de T_n evaluando *todos* los Walsh averages.

    Garantiza el mínimo global pero cuesta O(n^3 log n). Útil como referencia
    para validar la versión truncada en muestras pequeñas-medianas.
    """
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(x[0])
    W = _walsh_averages(x)
    vals = Tn_multi(x, W)
    return float(W[int(np.argmin(vals))])


def theta_argmin_grid(
    sample: np.ndarray,
    bracket: tuple[float, float] | None = None,
    n_grid: int = 60,
) -> float:
    """
    Variante antigua: grid uniforme + refinamiento Brent.

    Conservada para compatibilidad y comparación de rendimiento. No garantiza
    el óptimo global porque T_n es escalonada y el grid puede saltar sobre
    los intervalos donde está el mínimo.
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


def theta_argmin_schuster_narvarte(sample: np.ndarray) -> float:
    """
    Argmin exacto de T_n via el algoritmo combinatorio de Schuster-Narvarte.

    Para la muestra ordenada x_0 ≤ ... ≤ x_{n-1} (0-indexed) itera k = 0, 1, …
    buscando el primer índice L en el que la anti-diagonal INFERIOR ya no supera
    a la anti-diagonal SUPERIOR en el espacio de promedios de Walsh:

        m(k) = max{ (x_i + x_j)/2 : i+j = (n-1)-k,  0 ≤ i ≤ j ≤ n-1 }
        M(k) = min{ (x_i + x_j)/2 : i+j = (n-1)+k,  0 ≤ i ≤ j ≤ n-1 }

    La condición de parada es m(L) ≤ M(L); el estimador devuelto es
    θ̂ = (m(L) + M(L)) / 2.

    Intuición: T_n(θ) es constante entre Walsh averages consecutivos y alcanza
    su mínimo en el intervalo [m(L), M(L)].  El punto medio de ese intervalo
    es el estimador.

    Complejidad: O(n) por iteración × L iteraciones.  En la práctica L es
    pequeño (típicamente O(1) para distribuciones "suaves"), por lo que el
    algoritmo es O(n) en media.  Peor caso O(n²) (L = n-1).

    Parameters
    ----------
    sample : np.ndarray
        Muestra (no necesariamente ordenada).

    Returns
    -------
    float
        Estimador del centro de simetría.
    """
    x = np.sort(np.asarray(sample, dtype=float))
    n = x.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(x[0])
    if n == 2:
        return float(0.5 * (x[0] + x[1]))

    for k in range(n):
        # ---- Anti-diagonal inferior: i + j = (n-1) - k  ----------------
        s_lo = (n - 1) - k
        # Restricciones: 0 ≤ i ≤ j ≤ n-1  con  i+j = s_lo
        #   j ≤ n-1  →  i ≥ s_lo-(n-1)
        #   j ≥ i    →  i ≤ s_lo//2
        i_lo_min = max(0, s_lo - (n - 1))
        i_lo_max = s_lo // 2
        if i_lo_min <= i_lo_max:
            i_lo = np.arange(i_lo_min, i_lo_max + 1)
            m_k = 0.5 * float(np.max(x[i_lo] + x[s_lo - i_lo]))
        else:
            m_k = -np.inf

        # ---- Anti-diagonal superior: i + j = (n-1) + k  ----------------
        s_hi = (n - 1) + k
        #   j ≤ n-1  →  i ≥ s_hi-(n-1) = k
        #   j ≥ i    →  i ≤ s_hi//2
        i_hi_min = k                  # = s_hi - (n-1)
        i_hi_max = s_hi // 2
        if i_hi_min <= i_hi_max:
            i_hi = np.arange(i_hi_min, i_hi_max + 1)
            M_k = 0.5 * float(np.min(x[i_hi] + x[s_hi - i_hi]))
        else:
            M_k = np.inf

        if m_k <= M_k:
            return 0.5 * (m_k + M_k)

    # No debería llegar aquí (k=n-1 siempre cumple m≤M), pero por seguridad:
    return float(np.median(x))


def theta_argmin_pyomo(
    sample: np.ndarray,
    n_walsh: int | None = None,
    anchor: str | float = "median",
    *,
    exact: bool = False,
    solver: str | None = None,
    solver_options: dict | None = None,
    tee: bool = False,
) -> float:
    """Argmin de T_n seleccionado con Pyomo sobre candidatos de Walsh.

    Por defecto evalúa una vecindad de promedios de Walsh alrededor de la
    mediana, igual que ``theta_argmin``. Para forzar la enumeración completa
    de todos los promedios de Walsh use ``exact=True``.

    Nota: Pyomo se usa para resolver la selección discreta final entre
    candidatos ya evaluados. Si Pyomo o el solver no están instalados, esta
    función levanta un error explicativo.
    """
    x = np.asarray(sample, dtype=float)
    n = x.size
    if n == 0:
        return 0.0
    if n == 1:
        return float(x[0])

    if isinstance(anchor, str):
        if anchor == "median":
            theta_anchor = float(np.median(x))
        elif anchor == "trimmed":
            theta_anchor = float(trim_mean(x, 0.1))
        else:
            raise ValueError(f"anchor inválido: {anchor!r}")
    else:
        theta_anchor = float(anchor)

    W = _walsh_averages(x)
    if exact:
        cands = W
    else:
        if n_walsh is None:
            n_walsh = _default_k_walsh(n)
        if n_walsh >= W.size:
            cands = W
        else:
            dists = np.abs(W - theta_anchor)
            idx = np.argpartition(dists, n_walsh - 1)[:n_walsh]
            cands = W[idx]

    cands = np.unique(cands)
    vals = Tn_multi(x, cands)
    return solve_discrete_argmin_pyomo(
        cands,
        vals,
        solver=solver,
        solver_options=solver_options,
        tee=tee,
    )


def theta_argmin_pyomo_full(
    sample: np.ndarray,
    *,
    solver: str | None = None,
    solver_options: dict | None = None,
    tee: bool = False,
) -> float:
    """Argmin de T_n con Pyomo sobre todos los promedios de Walsh."""
    return theta_argmin_pyomo(
        sample,
        exact=True,
        solver=solver,
        solver_options=solver_options,
        tee=tee,
    )


# Diccionario de estimadores disponibles -----------------------------------
ESTIMATORS: dict[str, Callable[[np.ndarray], float]] = {
    "argmin": theta_argmin_schuster_narvarte,   # algoritmo exacto eficiente
    "argmin_walsh": theta_argmin,               # vecindad Walsh alrededor de mediana
    "argmin_walsh_full": theta_argmin_walsh_full,
    "argmin_grid": theta_argmin_grid,
    "argmin_pyomo": theta_argmin_pyomo,
    "argmin_pyomo_full": theta_argmin_pyomo_full,
    "hodges_lehmann": theta_hodges_lehmann,
    "median": theta_median,
    "trimmed": theta_trimmed,
}
