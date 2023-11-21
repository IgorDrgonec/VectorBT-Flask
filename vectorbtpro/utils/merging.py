# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Utilities for merging."""

import attr
from functools import partial

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.annotations import get_annotations, Annotatable, A
from vectorbtpro.utils.template import substitute_templates
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "MergeFunc",
]

__pdoc__ = {}


MergeFuncT = tp.TypeVar("MergeFuncT", bound="MergeFunc")


@attr.s(frozen=True)
class MergeFunc(Annotatable):
    """Class representing a merging function and its keyword arguments.

    Can be directly called to call the underlying (already resolved and with keyword
    arguments attached) merging function."""

    merge_func: tp.MergeFuncLike = attr.ib()
    """Merging function.
    
    Can be a name or a callable."""

    merge_kwargs: tp.KwargsLike = attr.ib(default=None)
    """Keyword arguments passed to the merging function."""

    context: tp.KwargsLike = attr.ib(default=None)
    """Context for substituting templates in `MergeFunc.merge_func` and `MergeFunc.merge_kwargs`."""

    sub_id_prefix: str = attr.ib(default="")
    """Prefix for the substitution id."""

    def evolve(self: MergeFuncT, merge_kwargs: tp.KwargsLike = None, context: tp.KwargsLike = None) -> MergeFuncT:
        """Evolve the instance with new keyword arguments and context."""
        merge_kwargs = merge_dicts(self.merge_kwargs, merge_kwargs)
        context = merge_dicts(self.context, context)
        return attr.evolve(self, merge_kwargs=merge_kwargs, context=context)

    def resolve_merge_func(self) -> tp.Optional[tp.Callable]:
        """Get the merging function where keyword arguments are hard-coded."""
        from vectorbtpro.base.merging import resolve_merge_func

        merge_func = resolve_merge_func(self.merge_func)
        if merge_func is None:
            return None
        merge_kwargs = self.merge_kwargs
        if merge_kwargs is None:
            merge_kwargs = {}
        merge_func = substitute_templates(merge_func, self.context, sub_id=self.sub_id_prefix + "merge_func")
        merge_kwargs = substitute_templates(merge_kwargs, self.context, sub_id=self.sub_id_prefix + "merge_kwargs")
        return partial(merge_func, **merge_kwargs)

    def __call__(self, *objs, **kwargs) -> tp.Any:
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        merge_func = self.resolve_merge_func()
        if merge_func is None:
            return objs
        return merge_func(objs, **kwargs)


def parse_merge_func(func: tp.Callable) -> tp.Optional[MergeFunc]:
    """Parser the merging function from the function's annotations."""
    annotations = get_annotations(func)
    merge_func = None
    for k, v in annotations.items():
        if k == "return":
            if not isinstance(v, A):
                v = A(v)
            for obj in v.get_objs():
                if isinstance(obj, str):
                    from vectorbtpro.base.merging import merge_func_config

                    if obj in merge_func_config:
                        obj = MergeFunc(obj)
                if checks.is_complex_sequence(obj):
                    for o in obj:
                        if o is None or isinstance(o, (str, MergeFunc)):
                            if merge_func is None:
                                merge_func = []
                            elif not isinstance(merge_func, list):
                                raise ValueError(f"Two merging functions found in annotations: {merge_func} and {o}")
                            merge_func.append(o)
                elif isinstance(obj, MergeFunc):
                    if merge_func is not None:
                        raise ValueError(f"Two merging functions found in annotations: {merge_func} and {obj}")
                    merge_func = obj
    return merge_func
