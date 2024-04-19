# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for showing progress bars."""

from collections import OrderedDict
from numbers import Number
from functools import wraps
from tqdm.std import tqdm

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.attr_ import MISSING

__all__ = [
    "ProgressBar",
    "ProgressHidden",
    "with_progress_hidden",
    "ProgressShown",
    "with_progress_shown",
]


ProgressBarT = tp.TypeVar("ProgressBarT", bound="ProgressBar")


class ProgressBar:
    """Context manager to manage a progress bar.

    Supported types:

    * 'tqdm_auto'
    * 'tqdm_notebook'
    * 'tqdm_gui'
    * 'tqdm'

    For defaults, see `vectorbtpro._settings.pbar`."""

    def __init__(
        self,
        *args,
        pbar_type: tp.Optional[str] = None,
        show_progress: tp.Optional[bool] = None,
        show_progress_desc: tp.Optional[bool] = None,
        desc_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        if pbar_type is None:
            pbar_type = pbar_cfg["type"]
        if pbar_type.lower() == "tqdm_auto":
            from tqdm.auto import tqdm as pbar_type
        elif pbar_type.lower() == "tqdm_notebook":
            from tqdm.notebook import tqdm as pbar_type
        elif pbar_type.lower() == "tqdm_gui":
            from tqdm.gui import tqdm as pbar_type
        elif pbar_type.lower() == "tqdm":
            from tqdm import tqdm as pbar_type
        else:
            raise ValueError(f"pbar_type cannot be '{pbar_type}'")
        if pbar_cfg["disable"]:
            show_progress = False
        if show_progress is None:
            show_progress = True
        kwargs = merge_dicts(pbar_cfg["kwargs"], kwargs)
        if pbar_cfg["disable_desc"]:
            show_progress_desc = False
        if show_progress_desc is None:
            show_progress_desc = True
        desc_kwargs = merge_dicts(pbar_cfg["desc_kwargs"], desc_kwargs)

        self._pbar_type = pbar_type
        self._show_progress = show_progress
        self._args = args
        self._kwargs = kwargs
        self._show_progress_desc = show_progress_desc
        self._desc_kwargs = desc_kwargs
        self._pbar = None

    @property
    def pbar_type(self) -> tp.Type[tqdm]:
        """Progess bar type."""
        return self._pbar_type

    @property
    def show_progress(self) -> bool:
        """Whether show the progress bar."""
        return self._show_progress

    @property
    def args(self) -> tp.Args:
        """Positional arguments passed to the progress bar."""
        return self._args

    @property
    def kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to the progress bar."""
        return self._kwargs

    @property
    def show_progress_desc(self) -> bool:
        """Whether show the progress bar description."""
        return self._show_progress_desc

    @property
    def desc_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `ProgressBar.set_description`."""
        return self._desc_kwargs

    @property
    def pbar(self) -> tp.Optional[tqdm]:
        """Progress bar."""
        return self._pbar

    def prepare_desc(self, desc: tp.Union[None, str, dict]) -> str:
        """Prepare description."""
        if desc is None or desc is MISSING:
            return ""
        if isinstance(desc, dict):
            new_desc = []
            for k, v in desc.items():
                if v is MISSING:
                    continue
                if isinstance(v, Number):
                    v = self.pbar.format_num(v)
                if not isinstance(v, str):
                    v = str(v)
                v = v.strip()
                if k is None or k is MISSING:
                    new_desc.append(v)
                else:
                    new_desc.append(k + "=" + v)
            return ', '.join(new_desc)
        return str(desc)

    def set_prefix(self, desc: tp.Union[None, str, dict], refresh: tp.Optional[bool] = None) -> None:
        """Set prefix.

        Prepares it with `ProgressBar.prepare_desc`."""
        if self.show_progress_desc:
            desc = self.prepare_desc(desc)
            if refresh is None:
                refresh = self.desc_kwargs.get("refresh", True)
            self.pbar.set_description_str(desc, refresh=refresh)

    def set_prefix_str(self, desc: str, refresh: tp.Optional[bool] = None) -> None:
        """Set prefix without preparation."""
        if self.show_progress_desc:
            if refresh is None:
                refresh = self.desc_kwargs.get("refresh", True)
            self.pbar.set_description_str(desc, refresh=refresh)

    def set_postfix(self, desc: tp.Union[None, str, dict], refresh: tp.Optional[bool] = None) -> None:
        """Set postfix.

        Prepares it with `ProgressBar.prepare_desc`."""
        if self.show_progress_desc:
            desc = self.prepare_desc(desc)
            if refresh is None:
                refresh = self.desc_kwargs.get("refresh", True)
            self.pbar.set_postfix_str(desc, refresh=refresh)

    def set_postfix_str(self, desc: str, refresh: tp.Optional[bool] = None) -> None:
        """Set postfix without preparation."""
        if self.show_progress_desc:
            if refresh is None:
                refresh = self.desc_kwargs.get("refresh", True)
            self.pbar.set_postfix_str(desc, refresh=refresh)

    def set_description(
        self,
        desc: tp.Union[None, str, dict],
        as_postfix: tp.Optional[bool] = None,
        refresh: tp.Optional[bool] = None,
    ) -> None:
        """Set description.

        Uses the method `ProgressBar.set_prefix` if `as_postfix=True` in `ProgressBar.desc_kwargs`.
        Otherwise, uses the method `ProgressBar.set_postfix`.

        Uses `ProgressBar.desc_kwargs` as keyword arguments."""
        if self.show_progress_desc:
            if as_postfix is None:
                as_postfix = self.desc_kwargs.get("as_postfix", True)
            if as_postfix:
                self.set_postfix(desc, refresh=refresh)
            else:
                self.set_prefix(desc, refresh=refresh)

    def set_description_str(
        self,
        desc: str,
        as_postfix: tp.Optional[bool] = None,
        refresh: tp.Optional[bool] = None,
    ) -> None:
        """Set description without preparation.

        Uses the method `ProgressBar.set_prefix_str` if `as_postfix=True` in `ProgressBar.desc_kwargs`.
        Otherwise, uses the method `ProgressBar.set_postfix_str`.

        Uses `ProgressBar.desc_kwargs` as keyword arguments."""
        if self.show_progress_desc:
            if as_postfix is None:
                as_postfix = self.desc_kwargs.get("as_postfix", True)
            if as_postfix:
                self.set_postfix_str(desc, refresh=refresh)
            else:
                self.set_prefix_str(desc, refresh=refresh)

    def __bool__(self) -> bool:
        return self.pbar.__bool__()

    def __len__(self) -> int:
        return self.pbar.__len__()

    def __reversed__(self) -> tp.Iterator:
        return self.pbar.__reversed__()

    def __contains__(self, item: tp.Any) -> bool:
        return self.pbar.__contains__(item)

    def __enter__(self: ProgressBarT) -> ProgressBarT:
        self._pbar = self.pbar_type(*self.args, disable=not self.show_progress, **self.kwargs)
        return self

    def __exit__(self, *args) -> None:
        self.pbar.close()

    def __del__(self) -> None:
        return self.pbar.__del__()

    def __iter__(self) -> tp.Iterator:
        return self.pbar.__iter__()

    def __getattr__(self, k: str) -> tp.Any:
        pbar = object.__getattribute__(self, "pbar")
        return getattr(pbar, k)


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
