# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for showing progress bars."""

from functools import wraps

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "get_pbar",
    "ProgressHidden",
    "with_progress_hidden",
    "ProgressShown",
    "with_progress_shown",
]


def get_pbar(*args, pbar_type: tp.Optional[str] = None, show_progress: bool = True, **kwargs) -> object:
    """Get a `tqdm` progress bar.

    Supported types:

    * 'tqdm_auto'
    * 'tqdm_notebook'
    * 'tqdm_gui'
    * 'tqdm'

    For defaults, see `vectorbtpro._settings.pbar`."""

    from vectorbtpro._settings import settings

    pbar_cfg = settings["pbar"]

    if pbar_cfg["disable"]:
        show_progress = False
    if pbar_type is None:
        pbar_type = pbar_cfg["type"]
    kwargs = merge_dicts(pbar_cfg["kwargs"], kwargs)

    if pbar_type.lower() == "tqdm_auto":
        from tqdm.auto import tqdm as pbar
    elif pbar_type.lower() == "tqdm_notebook":
        from tqdm.notebook import tqdm as pbar
    elif pbar_type.lower() == "tqdm_gui":
        from tqdm.gui import tqdm as pbar
    elif pbar_type.lower() == "tqdm":
        from tqdm import tqdm as pbar
    else:
        raise ValueError(f"pbar_type cannot be '{pbar_type}'")
    return pbar(*args, disable=not show_progress, **kwargs)


def set_pbar_description(pbar: object, desc: tp.Union[None, str, dict], **desc_kwargs) -> None:
    """Set description, either as a string or dictionary."""
    if not pbar.disable and desc is not None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        if pbar_cfg["disable_desc"]:
            return
        desc_kwargs = merge_dicts(pbar_cfg["desc_kwargs"], desc_kwargs)
        as_postfix = desc_kwargs.pop("as_postfix", True)
        if isinstance(desc, dict):
            new_desc = []
            for k, v in desc.items():
                if k is None:
                    new_desc.append(str(v))
                else:
                    new_desc.append(str(k) + "=" + str(v))
            desc = ", ".join(new_desc)
        if as_postfix:
            pbar.set_postfix_str(desc, **desc_kwargs)
        else:
            pbar.set_description(desc, **desc_kwargs)


ProgressHiddenT = tp.TypeVar("ProgressHiddenT", bound="ProgressHidden")


class ProgressHidden:
    """Context manager to hide progress."""

    def __init__(self) -> None:
        self._init_disable = None

    @property
    def init_disable(self) -> tp.Optional[bool]:
        """Initial `disable` value."""
        return self._init_disable

    def __enter__(self: ProgressHiddenT) -> ProgressHiddenT:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        self._init_disable = pbar_cfg["disable"]
        pbar_cfg["disable"] = True

        return self

    def __exit__(self, *args) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        pbar_cfg["disable"] = self.init_disable


def with_progress_hidden(*args) -> tp.Callable:
    """Decorator to run a function with `ProgressHidden`."""

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with ProgressHidden():
                return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


ProgressShownT = tp.TypeVar("ProgressShownT", bound="ProgressShown")


class ProgressShown:
    """Context manager to show progress."""

    def __init__(self) -> None:
        self._init_disable = None

    @property
    def init_disable(self) -> tp.Optional[bool]:
        """Initial `disable` value."""
        return self._init_disable

    def __enter__(self: ProgressShownT) -> ProgressShownT:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        self._init_disable = pbar_cfg["disable"]
        pbar_cfg["disable"] = False

        return self

    def __exit__(self, *args) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        pbar_cfg["disable"] = self.init_disable


def with_progress_shown(*args) -> tp.Callable:
    """Decorator to run a function with `ProgressShown`."""

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with ProgressShown():
                return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
