# Copyright (c) 2021 Oleg Polakow. All rights reserved.

import importlib
import pkgutil

from vectorbtpro import _typing as tp
from vectorbtpro._settings import settings
from vectorbtpro._version import __version__

# Silence warnings
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)
warnings.filterwarnings(
    "ignore", message="The localize method is no longer necessary, as this time zone supports the fold attribute"
)

if settings["importing"]["recursive_import"]:
    from vectorbtpro.utils.module_ import check_installed

    def _recursive_import(package):
        if isinstance(package, str):
            package = importlib.import_module(package)
        if not hasattr(package, "__climb__"):
            package.__climb__ = []
        if not hasattr(package, "__dont_climb_from__"):
            package.__dont_climb_from__ = []
        if not hasattr(package, "__import_if_installed__"):
            package.__import_if_installed__ = {}
        blacklist = []
        for k, v in package.__import_if_installed__.items():
            if not check_installed(v) or not settings["importing"][v]:
                blacklist.append(k)

        for importer, mod_name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            relative_name = mod_name.split(".")[-1]
            if relative_name in blacklist:
                continue
            if is_pkg:
                module = _recursive_import(mod_name)
            else:
                module = importlib.import_module(mod_name)
            if not hasattr(module, "__climb__"):
                module.__climb__ = []
            if relative_name not in package.__dont_climb_from__:
                for k in module.__climb__:
                    if hasattr(package, k) and getattr(package, k) is not getattr(module, k):
                        raise ValueError(
                            f"Attempt to override '{k}' in '{package.__name__}' via climbing from '{mod_name}'"
                        )
                    setattr(package, k, getattr(module, k))
                    package.__climb__.append(k)
        return package

    _recursive_import(__name__)

    from vectorbtpro.generic import nb, enums
    from vectorbtpro.indicators import nb as ind_nb, enums as ind_enums
    from vectorbtpro.labels import nb as lab_nb, enums as lab_enums
    from vectorbtpro.portfolio import nb as pf_nb, enums as pf_enums
    from vectorbtpro.records import nb as rec_nb
    from vectorbtpro.returns import nb as ret_nb, enums as ret_enums
    from vectorbtpro.signals import nb as sig_nb, enums as sig_enums
    from vectorbtpro.utils import datetime_ as dt, datetime_nb as dt_nb

__pdoc__ = dict()
__pdoc__["_settings"] = True
__pdoc__["_opt_deps"] = True
