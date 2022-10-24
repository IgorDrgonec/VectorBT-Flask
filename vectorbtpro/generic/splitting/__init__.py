# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Modules for splitting."""

from vectorbtpro.generic.splitting.base import RelRange, GapRange, Takeable, Splitter, SKLSplitter
from vectorbtpro.utils.module_ import create__all__

__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
