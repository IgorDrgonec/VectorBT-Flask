# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for working with paths."""

from pathlib import Path
from itertools import islice

from vectorbtpro import _typing as tp


def check_mkdir(
    dir_path: tp.PathLike,
    mkdir: tp.Optional[bool] = None,
    mode: tp.Optional[int] = None,
    parents: tp.Optional[bool] = None,
    exist_ok: tp.Optional[bool] = None,
):
    """Check whether the path to a directory exists and create if it doesn't.

    For defaults, see `mkdir` in `vectorbtpro._settings.path`."""
    from vectorbtpro._settings import settings

    mkdir_cfg = settings["path"]["mkdir"]

    if mkdir is None:
        mkdir = mkdir_cfg["mkdir"]
    if mode is None:
        mode = mkdir_cfg["mode"]
    if parents is None:
        parents = mkdir_cfg["parents"]
    if exist_ok is None:
        exist_ok = mkdir_cfg["exist_ok"]

    dir_path = Path(dir_path)
    if dir_path.exists() and not dir_path.is_dir():
        raise TypeError(f"Path '{dir_path}' is not a directory")
    if not dir_path.exists() and not mkdir:
        raise ValueError(f"Path '{dir_path}' not exists. Pass mkdir=True to create parent directories.")
    dir_path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)


def tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
    sort: bool = True,
    space="    ",
    branch="│   ",
    tee="├── ",
    last="└── ",
) -> str:
    """Given a directory Path object print a visual tree structure.

    Inspired by this answer: https://stackoverflow.com/a/59109706"""
    dir_path = Path(dir_path)
    files = 0
    directories = 0

    def _inner(dir_path: Path, prefix: str = "", level: int = -1) -> tp.Generator[str, None, None]:
        nonlocal files, directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        if sort:
            contents = sorted(contents)
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from _inner(path, prefix=prefix + extension, level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    tree_str = dir_path.name
    iterator = _inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        tree_str += "\n" + line
    if next(iterator, None):
        tree_str += "\n" + f"... length_limit, {length_limit}, reached, counted:"
    tree_str += "\n" + f"\n{directories} directories" + (f", {files} files" if files else "")
    return tree_str
