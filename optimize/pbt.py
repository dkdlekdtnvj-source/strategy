"""Utilities for lightweight Population-Based Training style mutations."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from .search_spaces import SpaceSpec, mutate_parameters


def generate_pbt_mutations(
    seeds: Iterable[Dict[str, object]],
    space: SpaceSpec,
    scale: float = 0.1,
    epsilon: float = 0.1,
    rng: np.random.Generator | None = None,
) -> List[Dict[str, object]]:
    """Return a list of mutated parameter dictionaries for continued exploration."""

    rng = rng or np.random.default_rng()
    seeds = list(seeds)
    if not seeds:
        return []

    mutations: List[Dict[str, object]] = []
    for params in seeds:
        mutated = mutate_parameters(params, space, scale=scale, rng=rng)
        mutations.append(mutated)
        if rng.random() < epsilon:
            # Inject a random jump by mutating again with a larger scale.
            mutations.append(mutate_parameters(mutated, space, scale=scale * 2.0, rng=rng))
    return mutations


__all__ = ["generate_pbt_mutations"]
