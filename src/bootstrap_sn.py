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
- Para argmin se usa `_sn_boot_batch_argmin_secant`: precomputa (a_n, b_n,
  |c_n|, Φ_n) una vez para las B remuestras como tensores (B, K), luego corre
  el método de la secante vectorizado sobre B (el mismo algoritmo que L-BFGS-B
  reduce en 1D). ~2× más rápido que el grid vectorizado, ~8× que L-BFGS-B
  secuencial.
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
def _sn_boot_batch_argmin(
    Y: np.ndarray,
    w_fn: WeightFn,
    q: int,
    t_grid: np.ndarray,
    bracket: tuple[float, float],
    n_theta: int = 80,
) -> np.ndarray:
    """Evalúa argmin_θ S_n sobre B remuestras simultáneamente.

    Construye un grid de n_theta candidatos en ``bracket`` y toma el mínimo
    para cada remuestra. Vectoriza via tensor (bs, K, M) en mini-lotes.
    Análogo a `_sn_boot_batch` pero incluye la minimización sobre θ,
    evitando el bucle Python × B con llamadas individuales a L-BFGS-B.

    Parameters
    ----------
    Y : (B, n) array — remuestras bootstrap
    w_fn, q, t_grid : igual que _sn_boot_batch
    bracket : (lo, hi) intervalo que cubre el argmin para todas las Y_b
    n_theta : puntos en la grilla de theta

    Returns
    -------
    S_boot : (B,) array con min_θ S_n(Y_b; θ) para cada b
    """
    B_total, n = Y.shape
    K = t_grid.size
    w_vals = w_fn.fn(t_grid)                                     # (K,)
    theta_cands = np.linspace(bracket[0], bracket[1], n_theta)  # (M,)

    # Precomputa cos/sin del producto t × θ — constante para todo el batch
    tth = t_grid[:, None] * theta_cands[None, :]  # (K, M)
    cos_tth = np.cos(tth)                         # (K, M)
    sin_tth = np.sin(tth)                         # (K, M)

    # Mini-lotes: ≤ 200 MB por lote (bs × K × M × 8 bytes)
    B_sub = max(1, int(200_000_000 // (K * n_theta * 8)))
    S_boot = np.empty(B_total, dtype=float)

    for start in range(0, B_total, B_sub):
        end = min(start + B_sub, B_total)
        Yb = Y[start:end]                                    # (bs, n)
        bs = end - start

        arg = t_grid[None, :, None] * Yb[:, None, :]        # (bs, K, n)
        a_n = np.mean(np.cos(arg), axis=2)                   # (bs, K)
        b_n = np.mean(np.sin(arg), axis=2)                   # (bs, K)
        abs_cn = np.sqrt(a_n ** 2 + b_n ** 2)               # (bs, K)

        # A[b,k,m] = a_n[b,k] cos(t_k θ_m) + b_n[b,k] sin(t_k θ_m)
        A = (a_n[:, :, None] * cos_tth[None, :, :]
             + b_n[:, :, None] * sin_tth[None, :, :])        # (bs, K, M)

        diff = 2.0 * abs_cn[:, :, None] * (abs_cn[:, :, None] - A)
        np.maximum(diff, 0.0, out=diff)
        integrand = np.sqrt(diff) if q == 1 else diff        # (bs, K, M)

        S_vals = np.trapezoid(
            integrand * w_vals[None, :, None], t_grid, axis=1
        )                                                    # (bs, M)
        S_boot[start:end] = S_vals.min(axis=1)              # (bs,)

    return S_boot


def _sn_boot_batch_argmin_secant(
    Y: np.ndarray,
    w_fn: WeightFn,
    q: int,
    t_grid: np.ndarray,
    bracket: tuple[float, float],
    max_steps: int = 25,
    tol: float = 1e-5,
) -> np.ndarray:
    """Método de la secante batcheado para argmin S_n sobre B remuestras.

    Usa el gradiente analítico de S_n (exactamente las fórmulas del marco
    teórico) vectorizado sobre las B replicas simultáneamente.  En 1D,
    L-BFGS-B se reduce al método de la secante; aquí lo implementamos
    directamente para poder batchear sobre B.

    Flujo
    -----
    1. Precomputa (a_n, b_n, |c_n|, Φ_n) para las B remuestras → (B, K).
       (mini-lotes para controlar RAM durante el paso (B, K, n))
    2. Inicializa theta_b = mediana(Y_b) y un punto perturbado theta_prev.
    3. Itera la secante vectorizada:
           dg = grad(theta) - grad(theta_prev)
           theta_new = theta - grad(theta) * (theta - theta_prev) / dg
       con recorte al bracket cuando |dg| < eps (fallback gradiente).
    4. Evalúa S_n(Y_b; theta*_b) con los precomputes → (B,).

    Parameters
    ----------
    Y : (B, n) array
    bracket : (lo, hi) que acota el argmin para todas las Y_b
    max_steps : iteraciones máximas de la secante
    tol : tolerancia de convergencia (|Δθ| máx sobre B)

    Returns
    -------
    S_boot : (B,) con S_n(Y_b; argmin_θ S_n(Y_b)) para cada b
    """
    B_total, n = Y.shape
    K = t_grid.size
    w_vals = w_fn.fn(t_grid)                        # (K,)
    lo, hi = bracket
    half_width = 0.5 * (hi - lo)

    # ------------------------------------------------------------------
    # Paso 1: precomputa (a_n, b_n, abs_cn, Phi_n) → (B, K)
    # mini-lotes sobre B para manejar el tensor (bs, K, n)
    # ------------------------------------------------------------------
    B_sub = max(1, int(200_000_000 // (K * n * 8)))
    a_n_all   = np.empty((B_total, K), dtype=float)
    b_n_all   = np.empty((B_total, K), dtype=float)
    abs_cn_all = np.empty((B_total, K), dtype=float)

    for start in range(0, B_total, B_sub):
        end = min(start + B_sub, B_total)
        Yb  = Y[start:end]                                   # (bs, n)
        arg = t_grid[None, :, None] * Yb[:, None, :]        # (bs, K, n)
        a_n = np.mean(np.cos(arg), axis=2)                   # (bs, K)
        b_n = np.mean(np.sin(arg), axis=2)
        a_n_all[start:end]    = a_n
        b_n_all[start:end]    = b_n
        abs_cn_all[start:end] = np.sqrt(a_n ** 2 + b_n ** 2)

    Phi_n = np.arctan2(b_n_all, a_n_all)                    # (B, K)

    # ------------------------------------------------------------------
    # Gradiente vectorizado ∂S_n/∂θ para todos los B thetas a la vez
    # ------------------------------------------------------------------
    def _grad(th: np.ndarray) -> np.ndarray:
        """th : (B,) → grad : (B,)"""
        if q == 1:
            half = (Phi_n - t_grid[None, :] * th[:, None]) / 2.0   # (B, K)
            sin_h = np.sin(half)
            cos_h = np.cos(half)
            integ = (-t_grid[None, :] * abs_cn_all
                     * np.sign(sin_h) * cos_h * w_vals[None, :])
        else:
            tth = t_grid[None, :] * th[:, None]                     # (B, K)
            B_n = -a_n_all * np.sin(tth) + b_n_all * np.cos(tth)
            integ = -2.0 * t_grid[None, :] * abs_cn_all * B_n * w_vals[None, :]
        return np.trapezoid(integ, t_grid, axis=1)                   # (B,)

    # ------------------------------------------------------------------
    # Paso 2: inicialización
    # ------------------------------------------------------------------
    thetas    = np.median(Y, axis=1).clip(lo, hi)           # (B,)
    delta     = (0.05 * half_width)                         # perturbación fija
    th_prev   = (thetas - delta).clip(lo, hi)               # (B,)

    grad_curr = _grad(thetas)                               # (B,)
    grad_prev = _grad(th_prev)                              # (B,)

    # ------------------------------------------------------------------
    # Paso 3: iteración del método de la secante
    # ------------------------------------------------------------------
    for _ in range(max_steps):
        dg = grad_curr - grad_prev                          # (B,)
        dt = thetas - th_prev                               # (B,)

        # Secante cuando |dg| es suficiente; fallback gradiente puro si no
        safe = np.abs(dg) > 1e-12 * (np.abs(grad_curr) + 1e-30)
        with np.errstate(divide="ignore", invalid="ignore"):
            secant_step = grad_curr * dt / dg
        step = np.where(safe,
                        secant_step,
                        0.1 * half_width * np.sign(grad_curr + 1e-30))

        # Recortar paso al bracket
        step = np.clip(step, -(hi - lo), hi - lo)
        theta_new = np.clip(thetas - step, lo, hi)         # (B,)

        if np.max(np.abs(theta_new - thetas)) < tol:
            thetas = theta_new
            break

        th_prev   = thetas
        grad_prev = grad_curr
        thetas    = theta_new
        grad_curr = _grad(thetas)

    # ------------------------------------------------------------------
    # Paso 4: evalúa S_n(Y_b; theta*_b) con los precomputes
    # ------------------------------------------------------------------
    tth = t_grid[None, :] * thetas[:, None]                # (B, K)
    A   = a_n_all * np.cos(tth) + b_n_all * np.sin(tth)   # (B, K)
    diff = 2.0 * abs_cn_all * (abs_cn_all - A)
    np.maximum(diff, 0.0, out=diff)
    integrand = np.sqrt(diff) if q == 1 else diff
    return np.trapezoid(integrand * w_vals[None, :], t_grid, axis=1)  # (B,)


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
        # --- Secante batcheado: gradiente analítico vectorizado sobre B ---
        lo, hi = float(support.min()), float(support.max())
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        bracket = (lo - pad, hi + pad)
        S_boot = _sn_boot_batch_argmin_secant(Y_boot, w_fn, q, t_grid, bracket)

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
