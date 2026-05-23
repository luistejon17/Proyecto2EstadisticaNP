"""Tests unitarios para sF_n, T_n y estimadores del centro de simetría."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bootstrap_ks import bootstrap_test_Tn
from src.statistics_ks import (
    ESTIMATORS,
    Tn_statistic,
    symmetrized_sample,
    theta_argmin,
    theta_median,
    theta_trimmed,
)


def _Tn_reference(sample: np.ndarray, theta: float) -> float:
    """Implementación de referencia ingenua (O(n^2)) para verificar Tn_statistic."""
    x = np.asarray(sample, dtype=float)
    n = x.size
    refl = 2.0 * theta - x
    grid = np.unique(np.concatenate([x, refl]))
    best = 0.0
    for g in grid:
        Fn = np.mean(x <= g)
        Fn_minus = np.mean(x < (2.0 * theta - g))
        diff = 0.5 * (Fn - 1.0 + Fn_minus)
        if abs(diff) > best:
            best = abs(diff)
    return float(np.sqrt(n) * best)


def test_Tn_matches_reference():
    rng = np.random.default_rng(42)
    for theta_true in [0.0, 1.5, -2.0]:
        x = theta_true + rng.normal(size=30)
        for theta in np.linspace(theta_true - 1, theta_true + 1, 5):
            fast = Tn_statistic(x, theta)
            ref = _Tn_reference(x, theta)
            assert np.isclose(fast, ref, atol=1e-10), (theta_true, theta, fast, ref)


def test_Tn_vanishes_for_perfectly_symmetric_sample():
    """En una muestra exactamente simétrica respecto a theta, T_n(theta) = 0."""
    x = np.array([-2.0, -1.0, 0.5, 1.5, 1.0, -0.5])  # simétrica respecto a 0
    # Forzar simetría perfecta
    x_sym = np.concatenate([np.array([0.5, 1.0, 2.0]), -np.array([0.5, 1.0, 2.0])])
    assert np.isclose(Tn_statistic(x_sym, 0.0), 0.0, atol=1e-12)


def test_Tn_invariant_under_translation():
    """T_n((X + c); theta + c) = T_n(X; theta)."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=50)
    theta = 0.3
    c = 5.0
    assert np.isclose(
        Tn_statistic(x, theta), Tn_statistic(x + c, theta + c), atol=1e-12
    )


def test_theta_argmin_recovers_symmetry_center():
    """Para muestras grandes simétricas, argmin debe estar cerca del centro."""
    rng = np.random.default_rng(2026)
    centers = [0.0, 2.5, -1.5]
    for c in centers:
        x = c + rng.normal(size=400)
        est = theta_argmin(x)
        assert abs(est - c) < 0.25, f"argmin={est:.3f}, true={c}"


def test_theta_median_and_trimmed():
    rng = np.random.default_rng(11)
    x = rng.normal(loc=3.0, scale=1.0, size=500)
    assert abs(theta_median(x) - 3.0) < 0.2
    assert abs(theta_trimmed(x, trim=0.1) - 3.0) < 0.2


def test_symmetrized_sample_is_symmetric_around_theta():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    theta = 2.0
    s = symmetrized_sample(x, theta)
    # Para cualquier z, su reflejo 2*theta - z debe estar también en s
    for z in s:
        assert any(np.isclose(2 * theta - z, w) for w in s)


def test_bootstrap_under_h0_p_value_uniform():
    """Bajo H0 el p-valor debe distribuirse aproximadamente uniforme en (0,1).

    Verificamos que en promedio rechazamos ~ alpha = 0.05.
    """
    rng = np.random.default_rng(123)
    R = 150
    rejects = 0
    for _ in range(R):
        x = rng.normal(loc=2.0, scale=1.0, size=40)
        res = bootstrap_test_Tn(x, estimator="median", B=199, rng=rng)
        if res.p_value < 0.05:
            rejects += 1
    rate = rejects / R
    # Debería estar cerca de 0.05; tolerancia generosa por R modesto
    assert 0.0 <= rate <= 0.20, f"Tasa de rechazo bajo H0 = {rate:.3f}"


def test_bootstrap_has_power_against_gamma():
    """Bajo Ha (gamma asimétrica) debemos rechazar con alta probabilidad."""
    rng = np.random.default_rng(321)
    R = 60
    rejects = 0
    for _ in range(R):
        x = rng.gamma(2.0, 1.0, size=80)
        res = bootstrap_test_Tn(x, estimator="median", B=199, rng=rng)
        if res.p_value < 0.05:
            rejects += 1
    rate = rejects / R
    # Sanity check; el estudio MC formal dará la potencia real
    assert rate > 0.4, f"Potencia bajo Gamma demasiado baja = {rate:.3f}"


if __name__ == "__main__":
    failures = []
    tests = [
        ("Tn matches reference", test_Tn_matches_reference),
        ("Tn vanishes for symmetric", test_Tn_vanishes_for_perfectly_symmetric_sample),
        ("Tn translation invariance", test_Tn_invariant_under_translation),
        ("argmin recovers center", test_theta_argmin_recovers_symmetry_center),
        ("median/trimmed", test_theta_median_and_trimmed),
        ("symmetrized sample", test_symmetrized_sample_is_symmetric_around_theta),
        ("bootstrap H0 level", test_bootstrap_under_h0_p_value_uniform),
        ("bootstrap power", test_bootstrap_has_power_against_gamma),
    ]
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
