"""Utilidades opcionales para seleccionar argmins discretos con Pyomo.

Estas funciones no importan Pyomo al cargar el paquete. Pyomo solo se requiere
cuando se llama explícitamente a un estimador ``*_pyomo``.
"""
from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_PYOMO_SOLVERS = ("appsi_highs", "highs", "glpk", "cbc")


def solve_discrete_argmin_pyomo(
    candidates: np.ndarray,
    objective_values: np.ndarray,
    *,
    solver: str | None = None,
    solver_options: dict[str, Any] | None = None,
    tee: bool = False,
) -> float:
    """Selecciona el candidato con menor valor objetivo usando Pyomo.

    El modelo es un MILP trivial:

        min sum_i value_i y_i
        s.t. sum_i y_i = 1, y_i in {0, 1}.

    Parameters
    ----------
    candidates:
        Valores candidatos de theta.
    objective_values:
        Valor del estadístico para cada candidato.
    solver:
        Nombre del solver Pyomo. Si es None, se prueban algunos solvers
        comunes: appsi_highs, highs, glpk y cbc.
    solver_options:
        Opciones pasadas al solver seleccionado.
    tee:
        Si True, muestra la salida del solver.
    """
    theta = np.asarray(candidates, dtype=float).ravel()
    values = np.asarray(objective_values, dtype=float).ravel()

    if theta.size != values.size:
        raise ValueError("candidates y objective_values deben tener el mismo tamaño.")
    if theta.size == 0:
        raise ValueError("Se necesita al menos un candidato para theta.")

    mask = np.isfinite(theta) & np.isfinite(values)
    theta = theta[mask]
    values = values[mask]
    if theta.size == 0:
        raise ValueError("No hay candidatos finitos para optimizar con Pyomo.")
    if theta.size == 1:
        return float(theta[0])

    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError(
            "El estimador argmin_pyomo requiere instalar Pyomo. "
            "Instala, por ejemplo, `pyomo` y un solver compatible como `highspy`, "
            "o usa estimator='argmin'."
        ) from exc

    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, int(theta.size) - 1)
    model.y = pyo.Var(model.I, domain=pyo.Binary)
    model.choose_one = pyo.Constraint(
        expr=sum(model.y[i] for i in model.I) == 1
    )
    model.obj = pyo.Objective(
        expr=sum(float(values[i]) * model.y[i] for i in model.I),
        sense=pyo.minimize,
    )

    solver_names = (solver,) if solver is not None else DEFAULT_PYOMO_SOLVERS
    unavailable: list[str] = []
    last_error: Exception | None = None

    for solver_name in solver_names:
        opt = pyo.SolverFactory(solver_name)
        if not opt.available(exception_flag=False):
            unavailable.append(solver_name)
            continue
        if solver_options:
            for key, value in solver_options.items():
                opt.options[key] = value
        try:
            result = opt.solve(model, tee=tee)
        except Exception as exc:  # pragma: no cover - depende del solver local
            last_error = exc
            continue

        term = result.solver.termination_condition
        valid_terms = {pyo.TerminationCondition.optimal}
        for term_name in ("globallyOptimal", "locallyOptimal"):
            extra_term = getattr(pyo.TerminationCondition, term_name, None)
            if extra_term is not None:
                valid_terms.add(extra_term)
        if term not in valid_terms:
            last_error = RuntimeError(f"Pyomo terminó con condición {term}.")
            continue

        chosen = [
            i for i in range(theta.size)
            if pyo.value(model.y[i], exception=False) is not None
            and pyo.value(model.y[i]) > 0.5
        ]
        if chosen:
            return float(theta[chosen[0]])

        # Fallback defensivo si el solver no devuelve y_i claramente binario.
        y_vals = np.array([pyo.value(model.y[i], exception=False) or 0.0 for i in range(theta.size)])
        return float(theta[int(np.argmax(y_vals))])

    msg = (
        "No se pudo resolver el argmin con Pyomo. "
        f"Solvers no disponibles/probados: {', '.join(unavailable) or 'ninguno'}."
    )
    if last_error is not None:
        msg += f" Último error: {last_error}"
    msg += " Instala un solver como HiGHS (`highspy`) o GLPK/CBC."
    raise RuntimeError(msg)
