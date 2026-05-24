"""Tests unitarios para c_n, S_n y el bootstrap S_n."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap_sn import bootstrap_test_Sn
from src.characteristic_fn import (
    Sn_multi_theta,
    Sn_statistic,
    cauchy_weight,
    gaussian_weight,
    laplace_weight,
    make_t_grid,
    theta_argmin_Sn,
    theta_median,
    theta_trimmed,
    uniform_weight,
)


def _Sn_reference(sample, theta, w_fn, q, t_grid):
    """Implementación de referencia ingenua para verificar Sn_statistic."""
    x = np.asarray(sample, dtype=float)
    n = x.size
    A = np.array([np.mean(np.cos(t * (x - theta))) for t in t_grid])
    B = np.array([np.mean(np.sin(t * (x - theta))) for t in t_grid])
    abs_cn = np.sqrt(A ** 2 + B ** 2)
    diff_sq = (A - abs_cn) ** 2 + B ** 2
    integrand = diff_sq ** (q / 2.0)
    weights = w_fn.fn(t_grid)
    return float(np.trapezoid(integrand * weights, t_grid))


def test_Sn_matches_reference():
    """Versión vectorizada vs implementación ingenua."""
    rng = np.random.default_rng(42)
    w = gaussian_weight(1.0)
    t_grid = make_t_grid(w, n_points=151)
    for theta_true in [0.0, 1.5, -2.0]:
        x = theta_true + rng.normal(size=30)
        for q in [1, 2]:
            for theta in np.linspace(theta_true - 0.5, theta_true + 0.5, 4):
                fast = Sn_statistic(x, theta, w, q=q, t_grid=t_grid)
                slow = _Sn_reference(x, theta, w, q, t_grid)
                assert np.isclose(fast, slow, atol=1e-10), (q, theta, fast, slow)


def test_Sn_invariant_under_translation():
    """S_n(X+c; θ+c) = S_n(X; θ)."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=50)
    w = gaussian_weight(1.0)
    t_grid = make_t_grid(w)
    for q in [1, 2]:
        s1 = Sn_statistic(x, 0.3, w, q=q, t_grid=t_grid)
        s2 = Sn_statistic(x + 5.0, 5.3, w, q=q, t_grid=t_grid)
        assert np.isclose(s1, s2, atol=1e-10), (q, s1, s2)


def test_Sn_multi_matches_loop():
    """Sn_multi_theta debe coincidir con llamadas individuales."""
    rng = np.random.default_rng(33)
    x = rng.normal(loc=1.0, size=40)
    w = laplace_weight(1.0)
    t_grid = make_t_grid(w)
    thetas = np.array([0.5, 1.0, 1.5])
    multi = Sn_multi_theta(x, thetas, w_fn=w, q=2, t_grid=t_grid)
    single = np.array([Sn_statistic(x, th, w, q=2, t_grid=t_grid) for th in thetas])
    assert np.allclose(multi, single, atol=1e-12)


def test_theta_argmin_Sn_recovers_center():
    """Para muestras simétricas grandes, argmin debe estar cerca del centro."""
    rng = np.random.default_rng(2026)
    w = gaussian_weight(1.0)
    centers = [0.0, 2.5, -1.5]
    for c in centers:
        x = c + rng.normal(size=400)
        th = theta_argmin_Sn(x, w_fn=w, q=2)
        assert abs(th - c) < 0.3, f"argmin={th:.3f}, true={c}"


def test_estimators_simple():
    rng = np.random.default_rng(11)
    x = rng.normal(loc=3.0, scale=1.0, size=500)
    assert abs(theta_median(x) - 3.0) < 0.2
    assert abs(theta_trimmed(x, trim=0.1) - 3.0) < 0.2


def test_weight_functions_integrate_to_one_approx():
    """Las densidades w(t) deben integrar ~1 sobre el soporte usado."""
    for w in [gaussian_weight(1.0), gaussian_weight(0.5),
              laplace_weight(1.0), uniform_weight(5.0)]:
        t_grid = make_t_grid(w, n_points=401)
        integral = np.trapezoid(w.fn(t_grid), t_grid)
        assert abs(integral - 1.0) < 0.01, (w.name, integral)


def test_bootstrap_h0_level():
    """Bajo H0 la tasa de rechazo debe ser cercana a alpha (con tolerancia)."""
    rng = np.random.default_rng(123)
    w = gaussian_weight(1.0)
    R = 60
    rejects = 0
    for _ in range(R):
        x = rng.normal(loc=2.0, scale=1.0, size=40)
        res = bootstrap_test_Sn(
            x, w_fn=w, q=2, estimator="median", B=199, rng=rng,
        )
        if res.p_value < 0.05:
            rejects += 1
    rate = rejects / R
    assert 0.0 <= rate <= 0.20, f"Tasa H0 = {rate:.3f}"


def test_bootstrap_power_against_gamma():
    """Bajo Ha (gamma) la potencia debe ser razonable."""
    rng = np.random.default_rng(321)
    w = gaussian_weight(1.0)
    R = 30
    rejects = 0
    for _ in range(R):
        x = rng.gamma(2.0, 1.0, size=80)
        res = bootstrap_test_Sn(
            x, w_fn=w, q=2, estimator="median", B=199, rng=rng,
        )
        if res.p_value < 0.05:
            rejects += 1
    rate = rejects / R
    assert rate > 0.25, f"Potencia bajo Gamma = {rate:.3f}"


if __name__ == "__main__":
    tests = [
        ("Sn vs reference", test_Sn_matches_reference),
        ("Sn translation invariance", test_Sn_invariant_under_translation),
        ("Sn_multi vs loop", test_Sn_multi_matches_loop),
        ("argmin recovers center", test_theta_argmin_Sn_recovers_center),
        ("median/trimmed", test_estimators_simple),
        ("weights integrate to 1", test_weight_functions_integrate_to_one_approx),
        ("bootstrap H0 level", test_bootstrap_h0_level),
        ("bootstrap power gamma", test_bootstrap_power_against_gamma),
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    if failures:
        sys.exit(1)
    print("\nAll tests passed.")
