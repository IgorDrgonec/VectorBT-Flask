# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules that register objects across vectorbtpro."""

from vectorbtpro.registries.ca_registry import CAQuery, CAQueryDelegator
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted

__all__ = [
    'CAQuery',
    'CAQueryDelegator',
    'register_jitted',
    'register_chunkable'
]

__pdoc__ = {k: False for k in __all__}
