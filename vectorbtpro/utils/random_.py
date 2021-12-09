# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for random number generation."""

import random

import numpy as np

from vectorbtpro.jit_registry import register_jitted


@register_jitted(cache=True)
def set_seed_nb(seed: int) -> None:
    """Set seed in numba."""
    np.random.seed(seed)


def set_seed(seed: int) -> None:
    """Set seed."""
    random.seed(seed)
    np.random.seed(seed)
    set_seed_nb(seed)
