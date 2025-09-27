"""Helpers for translating YAML search spaces to Optuna."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import optuna

SpaceSpec = Dict[str, Dict[str, object]]


def build_space(space: SpaceSpec) -> SpaceSpec:
    return space


def sample_parameters(trial: optuna.Trial, space: SpaceSpec) -> Dict[str, object]:
    params: Dict[str, object] = {}
    for name, spec in space.items():
        dtype = spec["type"]
        if dtype == "int":
            params[name] = trial.suggest_int(name, int(spec["min"]), int(spec["max"]), step=int(spec.get("step", 1)))
        elif dtype == "float":
            params[name] = trial.suggest_float(name, float(spec["min"]), float(spec["max"]), step=float(spec.get("step", 0.1)))
        elif dtype == "bool":
            params[name] = trial.suggest_categorical(name, [True, False])
        elif dtype == "choice":
            values = spec.get("values") or spec.get("options")
            if not values:
                raise ValueError(f"Choice parameter '{name}' requires a non-empty 'values' list.")
            params[name] = trial.suggest_categorical(name, list(values))
        else:
            raise ValueError(f"Unsupported parameter type: {dtype}")
    return params


def grid_choices(space: SpaceSpec) -> Dict[str, List[object]]:
    grid: Dict[str, List[object]] = {}
    for name, spec in space.items():
        dtype = spec["type"]
        if dtype == "int":
            grid[name] = list(range(int(spec["min"]), int(spec["max"]) + 1, int(spec.get("step", 1))))
        elif dtype == "float":
            step = float(spec.get("step", 0.1))
            values = np.arange(float(spec["min"]), float(spec["max"]) + 1e-12, step)
            grid[name] = [round(val, 10) for val in values.tolist()]
        elif dtype == "bool":
            grid[name] = [True, False]
        elif dtype == "choice":
            values = spec.get("values") or spec.get("options")
            if not values:
                raise ValueError(f"Choice parameter '{name}' requires a non-empty 'values' list for grid sampling.")
            grid[name] = list(values)
        else:
            raise ValueError(f"Unsupported parameter type for grid: {dtype}")
    return grid


def _parameter_bounds(name: str, spec: Dict[str, object]) -> Tuple[object, object]:
    if spec["type"] in {"int", "float"}:
        return spec.get("min"), spec.get("max")
    return None, None


def clamp_parameter(name: str, value: object, spec: Dict[str, object]) -> object:
    """Clamp a value to the valid range for a given parameter specification."""

    dtype = spec["type"]
    if dtype == "int":
        low, high = _parameter_bounds(name, spec)
        return int(min(max(int(value), int(low)), int(high)))
    if dtype == "float":
        low, high = _parameter_bounds(name, spec)
        return float(min(max(float(value), float(low)), float(high)))
    if dtype == "choice":
        options = list(spec.get("values") or spec.get("options") or [])
        return value if value in options else options[0]
    if dtype == "bool":
        return bool(value)
    raise ValueError(f"Unsupported parameter type for clamp: {dtype}")


def mutate_parameters(
    params: Dict[str, object],
    space: SpaceSpec,
    scale: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Dict[str, object]:
    """Return a neighbour of ``params`` by applying bounded random mutations."""

    rng = rng or np.random.default_rng()
    mutated = dict(params)
    for name, spec in space.items():
        dtype = spec["type"]
        if dtype in {"int", "float"}:
            low, high = _parameter_bounds(name, spec)
            if low is None or high is None:
                continue
            span = float(high) - float(low)
            if span <= 0:
                continue
            noise = rng.normal(loc=0.0, scale=scale * span)
            mutated_val = float(params.get(name, low)) + noise
            if dtype == "int":
                mutated_val = round(mutated_val)
            mutated[name] = clamp_parameter(name, mutated_val, spec)
        elif dtype == "bool":
            if rng.random() < scale:
                mutated[name] = not bool(params.get(name, False))
        elif dtype == "choice":
            options = list(spec.get("values") or spec.get("options") or [])
            if not options:
                continue
            current = params.get(name)
            if rng.random() < scale or current not in options:
                mutated[name] = rng.choice(options)
    return mutated


__all__ = [
    "SpaceSpec",
    "build_space",
    "sample_parameters",
    "grid_choices",
    "mutate_parameters",
    "clamp_parameter",
]
