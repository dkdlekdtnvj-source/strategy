"""Helpers for translating YAML search spaces to Optuna."""
from __future__ import annotations

from typing import Dict, Iterable, List

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
        else:
            raise ValueError(f"Unsupported parameter type for grid: {dtype}")
    return grid
