"""
Generadores de muestras para el estudio de simulación.

Distribuciones bajo H0 (simétricas):
    - Uniforme(a, b) con centro de simetría theta = (a+b)/2.
    - Cauchy(loc, scale) con centro de simetría theta = loc.

Distribuciones bajo Ha (asimétricas):
    - Gamma(shape, scale).
    - Weibull(shape, scale).
    - Pareto(shape, scale).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.random import Generator


# ---------------------------------------------------------------------------
# Especificación de un escenario distribucional
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistSpec:
    """
    Especificación de una distribución para el estudio.

    Attributes
    ----------
    name : str
        Nombre corto para etiquetar resultados.
    sampler : callable
        Función ``sampler(n, rng) -> np.ndarray`` que devuelve una muestra de tamaño n.
    theta_true : float or None
        Centro de simetría verdadero bajo H0. Para distribuciones bajo Ha es None.
    under_h0 : bool
        Indica si la distribución es simétrica (H0 verdadera) o no.
    """
    name: str
    sampler: Callable[[int, Generator], np.ndarray]
    theta_true: float | None
    under_h0: bool


# ---------------------------------------------------------------------------
# Distribuciones bajo H0 (simétricas, NO centradas en 0)
# ---------------------------------------------------------------------------
def uniform_h0(a: float = 1.0, b: float = 3.0) -> DistSpec:
    """Uniforme en [a, b] (simétrica respecto a (a+b)/2)."""
    center = (a + b) / 2.0

    def sampler(n: int, rng: Generator) -> np.ndarray:
        return rng.uniform(a, b, size=n)

    return DistSpec(
        name=f"Uniforme({a},{b})",
        sampler=sampler,
        theta_true=center,
        under_h0=True,
    )


def cauchy_h0(loc: float = 2.0, scale: float = 1.0) -> DistSpec:
    """Cauchy con localización loc (simétrica respecto a loc)."""
    def sampler(n: int, rng: Generator) -> np.ndarray:
        return loc + scale * rng.standard_cauchy(size=n)

    return DistSpec(
        name=f"Cauchy(loc={loc},scale={scale})",
        sampler=sampler,
        theta_true=loc,
        under_h0=True,
    )


def normal_h0(loc: float = 0.0, scale: float = 1.0) -> DistSpec:
    """Normal estándar (referencia adicional bajo H0)."""
    def sampler(n: int, rng: Generator) -> np.ndarray:
        return rng.normal(loc, scale, size=n)

    return DistSpec(
        name=f"Normal(loc={loc},scale={scale})",
        sampler=sampler,
        theta_true=loc,
        under_h0=True,
    )


# ---------------------------------------------------------------------------
# Distribuciones bajo Ha (asimétricas)
# ---------------------------------------------------------------------------
def gamma_ha(shape: float = 2.0, scale: float = 1.0) -> DistSpec:
    """Gamma(shape, scale) asimétrica."""
    def sampler(n: int, rng: Generator) -> np.ndarray:
        return rng.gamma(shape, scale, size=n)

    return DistSpec(
        name=f"Gamma(k={shape},s={scale})",
        sampler=sampler,
        theta_true=None,
        under_h0=False,
    )


def weibull_ha(shape: float = 1.5, scale: float = 1.0) -> DistSpec:
    """Weibull(shape, scale) asimétrica."""
    def sampler(n: int, rng: Generator) -> np.ndarray:
        return scale * rng.weibull(shape, size=n)

    return DistSpec(
        name=f"Weibull(k={shape},s={scale})",
        sampler=sampler,
        theta_true=None,
        under_h0=False,
    )


def pareto_ha(shape: float = 3.0, scale: float = 1.0) -> DistSpec:
    """Pareto Tipo II (Lomax) con cola pesada y asimétrica.

    ``rng.pareto(a)`` devuelve Lomax con parámetro de forma a. Reescalamos por
    ``scale`` para controlar el orden de magnitud.
    """
    def sampler(n: int, rng: Generator) -> np.ndarray:
        return scale * rng.pareto(shape, size=n)

    return DistSpec(
        name=f"Pareto(a={shape},s={scale})",
        sampler=sampler,
        theta_true=None,
        under_h0=False,
    )


# ---------------------------------------------------------------------------
# Conveniencia: catálogos por defecto
# ---------------------------------------------------------------------------
def default_h0_specs() -> list[DistSpec]:
    """Catálogo por defecto de escenarios bajo H0."""
    return [uniform_h0(1.0, 3.0), cauchy_h0(2.0, 1.0)]


def default_ha_specs() -> list[DistSpec]:
    """Catálogo por defecto de escenarios bajo Ha."""
    return [gamma_ha(2.0, 1.0), weibull_ha(1.5, 1.0), pareto_ha(3.0, 1.0)]
