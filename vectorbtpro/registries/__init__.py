# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules that register objects across vectorbtpro."""

from vectorbtpro.registries.ca_registry import ca_registry, CAQuery, CAQueryDelegator
from vectorbtpro.registries.ch_registry import ch_registry, register_chunkable
from vectorbtpro.registries.jit_registry import jit_registry, register_jitted

__all__ = [
    'ca_registry',
    'CAQuery',
    'CAQueryDelegator',
    'ch_registry',
    'register_chunkable',
    'jit_registry',
    'register_jitted',
]

__pdoc__ = {k: False for k in __all__}
