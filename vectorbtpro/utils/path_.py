# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for working with paths."""

from vectorbtpro import _typing as tp


def check_mkdir(dir_path: tp.PathLike,
                mkdir: tp.Optional[bool] = None,
                mode: tp.Optional[int] = None,
                parents: tp.Optional[bool] = None,
                exist_ok: tp.Optional[bool] = None):
    """Check whether the path to a directory exists and create if it doesn't.

    For defaults, see `mkdir` in `vectorbtpro._settings.path`."""
    from vectorbtpro._settings import settings
    mkdir_cfg = settings['path']['mkdir']

    if mkdir is None:
        mkdir = mkdir_cfg['mkdir']
    if mode is None:
        mode = mkdir_cfg['mode']
    if parents is None:
        parents = mkdir_cfg['parents']
    if exist_ok is None:
        exist_ok = mkdir_cfg['exist_ok']

    if dir_path.exists() and not dir_path.is_dir():
        raise TypeError(f"Path '{dir_path}' is not a directory")
    if not dir_path.exists() and not mkdir:
        raise ValueError(f"Path '{dir_path}' not exists. "
                         f"Pass mkdir=True to create parent directories.")
    dir_path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
