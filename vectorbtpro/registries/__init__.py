# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules that register objects across vectorbtpro."""

from vectorbtpro.registries.ca_registry import ca_reg, CAQuery, CAQueryDelegator
from vectorbtpro.registries.ch_registry import ch_reg, register_chunkable
from vectorbtpro.registries.jit_registry import jit_reg, register_jitted
from vectorbtpro.utils.module_ import create__all__

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
