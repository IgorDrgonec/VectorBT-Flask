# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for annotations."""

import attr

from vectorbtpro import _typing as tp
from vectorbtpro.utils.decorators import class_or_instancemethod

__all__ = [
    "Annotatable",
    "A",
    "VarArgs",
    "VarKwargs",
]

__pdoc__ = {}

try:
    from inspect import get_annotations
except ImportError:
    import sys
    import types
    import functools

    def get_annotations(obj, *, globals=None, locals=None, eval_str=False):
        """A backport of Python 3.10's inspect.get_annotations() function.

        See https://github.com/python/cpython/blob/main/Lib/inspect.py"""
        if isinstance(obj, type):
            # class
            obj_dict = getattr(obj, "__dict__", None)
            if obj_dict and hasattr(obj_dict, "get"):
                ann = obj_dict.get("__annotations__", None)
                if isinstance(ann, types.GetSetDescriptorType):
                    ann = None
            else:
                ann = None

            obj_globals = None
            module_name = getattr(obj, "__module__", None)
            if module_name:
                module = sys.modules.get(module_name, None)
                if module:
                    obj_globals = getattr(module, "__dict__", None)
            obj_locals = dict(vars(obj))
            unwrap = obj
        elif isinstance(obj, types.ModuleType):
            # module
            ann = getattr(obj, "__annotations__", None)
            obj_globals = getattr(obj, "__dict__")
            obj_locals = None
            unwrap = None
        elif callable(obj):
            # this includes types.Function, types.BuiltinFunctionType,
            # types.BuiltinMethodType, functools.partial, functools.singledispatch,
            # "class funclike" from Lib/test/test_inspect... on and on it goes.
            ann = getattr(obj, "__annotations__", None)
            obj_globals = getattr(obj, "__globals__", None)
            obj_locals = None
            unwrap = obj
        else:
            raise TypeError(f"{obj!r} is not a module, class, or callable.")

        if ann is None:
            return {}

        if not isinstance(ann, dict):
            raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

        if not ann:
            return {}

        if not eval_str:
            return dict(ann)

        if unwrap is not None:
            while True:
                if hasattr(unwrap, "__wrapped__"):
                    unwrap = unwrap.__wrapped__
                    continue
                if isinstance(unwrap, functools.partial):
                    unwrap = unwrap.func
                    continue
                break
            if hasattr(unwrap, "__globals__"):
                obj_globals = unwrap.__globals__

        if globals is None:
            globals = obj_globals
        if locals is None:
            locals = obj_locals

        return_value = {
            key: value if not isinstance(value, str) else eval(value, globals, locals) for key, value in ann.items()
        }
        return return_value


class MetaAnnotatable(type):
    """Metaclass that can be used in annotations."""

    def __or__(cls, other: tp.Any) -> "A":
        return A(cls, other)


class Annotatable(metaclass=MetaAnnotatable):
    """Class that can be used in annotations."""

    def __or__(self, other: tp.Any) -> "A":
        return A(self, other)


def has_annotatables(func: tp.Callable, target_cls=Annotatable) -> bool:
    """Check if a function has subclasses or instances of `Annotatable` in its signature."""
    annotations = get_annotations(func)
    for k, v in annotations.items():
        if isinstance(v, type) and issubclass(v, target_cls):
            return True
        if not isinstance(v, type) and isinstance(v, target_cls):
            return True
    return False


@attr.s(frozen=True, init=False)
class A(Annotatable):
    """Class representing an annotation consisting of one to multiple objects."""

    objs: tp.Tuple[object] = attr.ib()
    """Annotation objects."""

    def __init__(self, *objs) -> None:
        self.__attrs_init__(objs=objs)

    @class_or_instancemethod
    def get_objs(cls_or_self, objs: tp.Optional[tp.Tuple[object, ...]] = None) -> tp.Tuple[object, ...]:
        """Get a (flattened) list of objects."""
        if objs is None:
            if isinstance(cls_or_self, type):
                raise ValueError("Must provide objs")
            objs = cls_or_self.objs
        new_objs = []
        for obj in objs:
            if isinstance(obj, A):
                new_objs.extend(obj.get_objs())
            else:
                new_objs.append(obj)
        return tuple(new_objs)

    def __or__(self, other: tp.Any) -> "A":
        if isinstance(other, type(self)):
            return type(self)(self, other)
        return A(self, other)


@attr.s(frozen=True, init=False)
class VarArgs(Annotatable):
    """Class representing an annotation for variable positional arguments."""

    args: tp.Args = attr.ib()
    """Annotation objects."""

    def __init__(self, *args) -> None:
        self.__attrs_init__(args=args)


@attr.s(frozen=True, init=False)
class VarKwargs(Annotatable):
    """Class representing an annotation for variable keyword arguments."""

    kwargs: tp.Kwargs = attr.ib()
    """Annotation objects."""

    def __init__(self, **kwargs) -> None:
        self.__attrs_init__(kwargs=kwargs)
