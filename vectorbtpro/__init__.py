# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Next-gen framework for backtesting, algorithmic trading, and research."""

__pdoc__ = {}

# Import version
from vectorbtpro._version import __version__ as _version

__version__ = _version

# Most important classes
from vectorbtpro.utils import *
from vectorbtpro.base import *
from vectorbtpro.data import *
from vectorbtpro.generic import *
from vectorbtpro.indicators import *
from vectorbtpro.signals import *
from vectorbtpro.records import *
from vectorbtpro.portfolio import *
from vectorbtpro.labels import *
from vectorbtpro.messaging import *

# Most important modules and objects
from vectorbtpro import _typing as tp
from vectorbtpro._settings import settings
from vectorbtpro.jit_registry import register_jitted
from vectorbtpro.ch_registry import register_chunkable
from vectorbtpro.ca_registry import CAQuery, CAQueryDelegator
from vectorbtpro.jit_registry import register_jitted
from vectorbtpro.root_accessors import pd_acc, sr_acc, df_acc

# Import all submodules
from vectorbtpro.utils.module_ import import_submodules

# silence NumbaExperimentalFeatureWarning
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

__blacklist__ = []

try:
    import plotly
except ImportError:
    __blacklist__.append('px_accessors')

import_submodules(__name__)

__pdoc__['_settings'] = True
