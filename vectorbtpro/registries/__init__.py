# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules that register objects across vectorbtpro."""

from vectorbtpro.registries.ca_registry import ca_reg, CAQuery, CAQueryDelegator
from vectorbtpro.registries.ch_registry import ch_reg, register_chunkable
from vectorbtpro.registries.jit_registry import jit_reg, register_jitted

__all__ = [
    'ca_reg',
    'ch_reg',
    'jit_reg',
    'CAQuery',
    'CAQueryDelegator',
    'register_chunkable',
    'register_jitted',
]

__pdoc__ = {k: False for k in __all__}
