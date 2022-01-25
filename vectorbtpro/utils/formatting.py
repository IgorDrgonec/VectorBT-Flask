# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for formatting."""

import attr
import numpy as np

from vectorbtpro import _typing as tp


class Prettified:
    """Abstract class that can be prettified."""

    def prettify(self, **kwargs) -> str:
        """Prettify this object.

        !!! warning
            Calling `prettify` can lead to an infinite recursion.
            Make sure to pre-process this object."""
        raise NotImplementedError

    def __str__(self) -> str:
        try:
            return self.prettify()
        except NotImplementedError:
            return repr(self)


def prettify_dict(obj: tp.Any,
                  replace: tp.DictLike = None,
                  path: str = None,
                  htchar: str = '    ',
                  lfchar: str = '\n',
                  indent: int = 0) -> tp.Any:
    """Prettify a dictionary."""
    items = []
    for k, v in obj.items():
        if replace is None:
            replace = {}
        if path is None:
            new_path = k
        else:
            new_path = str(path) + '.' + str(k)
        if new_path in replace:
            new_v = replace[new_path]
        else:
            new_v = prettify(
                v,
                replace=replace,
                path=new_path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1
            )
        items.append(lfchar + htchar * (indent + 1) + repr(k) + ': ' + new_v)
    if type(obj) is dict:
        if len(items) == 0:
            return "{}"
        return "{%s}" % (','.join(items) + lfchar + htchar * indent)
    if len(items) == 0:
        return "%s({})" % (type(obj).__name__,)
    return "%s({%s})" % (type(obj).__name__, ','.join(items) + lfchar + htchar * indent)


def prettify_inited(cls: type,
                    kwargs: tp.Any,
                    replace: tp.DictLike = None,
                    path: str = None,
                    htchar: str = '    ',
                    lfchar: str = '\n',
                    indent: int = 0) -> tp.Any:
    """Prettify an instance initialized with keyword arguments."""
    items = []
    for k, v in kwargs.items():
        if replace is None:
            replace = {}
        if path is None:
            new_path = k
        else:
            new_path = str(path) + '.' + str(k)
        if new_path in replace:
            new_v = replace[new_path]
        else:
            new_v = prettify(
                v,
                replace=replace,
                path=new_path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1
            )
        k_repr = repr(k)
        if isinstance(k, str):
            k_repr = k_repr[1:-1]
        items.append(lfchar + htchar * (indent + 1) + k_repr + '=' + new_v)
    if len(items) == 0:
        return '%s()' % (cls.__name__,)
    return '%s(%s)' % (cls.__name__, ','.join(items) + lfchar + htchar * indent)


def prettify(obj: tp.Any,
             replace: tp.DictLike = None,
             path: str = None,
             htchar: str = '    ',
             lfchar: str = '\n',
             indent: int = 0) -> tp.Any:
    """Prettify an object.

    Unfolds regular Python data structures such as lists and tuples.

    If `obj` is an instance of `Prettified`, calls `Prettified.prettify`."""
    if isinstance(obj, Prettified):
        return obj.prettify(
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent
        )
    if attr.has(type(obj)):
        return prettify_inited(
            type(obj),
            attr.asdict(obj),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent
        )
    if isinstance(obj, dict):
        return prettify_dict(
            obj,
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent
        )
    if isinstance(obj, tuple) and hasattr(obj, '_asdict'):
        return prettify_inited(
            type(obj),
            obj._asdict(),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent
        )
    if isinstance(obj, (tuple, list, set, frozenset)):
        items = []
        for v in obj:
            new_v = prettify(
                v,
                replace=replace,
                path=path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent + 1
            )
            items.append(lfchar + htchar * (indent + 1) + new_v)
        if type(obj) is tuple:
            if len(items) == 0:
                return "()"
            return "(%s)" % (','.join(items) + lfchar + htchar * indent)
        if type(obj) is list:
            if len(items) == 0:
                return "[]"
            return "[%s]" % (','.join(items) + lfchar + htchar * indent)
        if type(obj) is set:
            if len(items) == 0:
                return "set()"
            return "{%s}" % (','.join(items) + lfchar + htchar * indent)
        if len(items) == 0:
            return "%s([])" % (type(obj).__name__,)
        return "%s([%s])" % (type(obj).__name__, ','.join(items) + lfchar + htchar * indent)
    if isinstance(obj, np.dtype) and hasattr(obj, "fields"):
        items = []
        for k, v in dict(obj.fields).items():
            items.append(lfchar + htchar * (indent + 1) + repr((k, str(v[0]))))
        return "np.dtype([%s])" % (','.join(items) + lfchar + htchar * indent)
    if hasattr(obj, 'shape') and isinstance(obj.shape, tuple) and len(obj.shape) > 0:
        module = type(obj).__module__
        qualname = type(obj).__qualname__
        return "<%s.%s object at %s of shape %s>" % (module, qualname, str(hex(id(obj))), obj.shape)
    if isinstance(obj, float):
        if np.isnan(obj):
            return "np.nan"
        if np.isposinf(obj):
            return "np.inf"
        if np.isneginf(obj):
            return "-np.inf"
    return repr(obj)
