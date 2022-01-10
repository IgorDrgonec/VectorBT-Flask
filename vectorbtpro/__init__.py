# Copyright (c) 2021 Oleg Polakow. All rights reserved.

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
from vectorbtpro.registries import *
from vectorbtpro.ohlcv import *
from vectorbtpro.px import *

# Most important modules and objects
from vectorbtpro import _typing as tp
from vectorbtpro._settings import settings
from vectorbtpro.accessors import (
    Vbt_Accessor as pd_acc,
    Vbt_SRAccessor as sr_acc,
    Vbt_DFAccessor as df_acc
)
from vectorbtpro.generic import nb

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
