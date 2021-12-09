# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for parsing."""

import ast
import attr
import inspect
import re
import sys

from vectorbtpro import _typing as tp


@attr.s(frozen=True)
class Regex:
    """Class for matching a regular expression."""

    pattern: str = attr.ib()
    """Pattern."""

    flags: int = attr.ib(default=0)
    """Flags."""

    def matches(self, string: str) -> bool:
        """Return whether the string matches the regular expression pattern."""
        return re.match(self.pattern, string, self.flags) is not None


def get_func_kwargs(func: tp.Callable) -> dict:
    """Get keyword arguments with defaults of a function."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_func_arg_names(func: tp.Callable, arg_kind: tp.Optional[tp.MaybeTuple[int]] = None) -> tp.List[str]:
    """Get argument names of a function."""
    signature = inspect.signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    if arg_kind is None:
        return [
            p.name for p in signature.parameters.values()
            if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD
        ]
    return [
        p.name for p in signature.parameters.values()
        if p.kind in arg_kind
    ]


def annotate_args(func: tp.Callable, args: tp.Args, kwargs: tp.Kwargs, only_passed: bool = False) -> tp.AnnArgs:
    """Annotate arguments and keyword arguments using the function's signature."""
    kwargs = dict(kwargs)
    signature = inspect.signature(func)
    signature.bind(*args, **kwargs)
    ann_args = dict()
    arg_i = 0

    for p in signature.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            if p.kind != p.KEYWORD_ONLY and arg_i < len(args):
                # Either positional-only arguments or keyword arguments passed as such
                ann_args[p.name] = dict(kind=p.kind, value=args[arg_i])
            else:
                # Either keyword-only arguments or positional arguments passed as such
                if p.name in kwargs:
                    ann_args[p.name] = dict(kind=p.kind, value=kwargs.pop(p.name))
                elif not only_passed:
                    ann_args[p.name] = dict(kind=p.kind, value=p.default)
            arg_i += 1
        elif p.kind == p.VAR_POSITIONAL:
            # *args
            if not only_passed or len(args[arg_i:]) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=args[arg_i:])
        else:
            # **kwargs
            if not only_passed or len(kwargs) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs)
    return ann_args


def flatten_ann_args(ann_args: tp.AnnArgs) -> tp.FlatAnnArgs:
    """Flatten annotated arguments."""
    flat_ann_args = []
    for arg_name, ann_arg in ann_args.items():
        if ann_arg['kind'] == inspect.Parameter.VAR_POSITIONAL:
            for v in ann_arg['value']:
                flat_ann_args.append(dict(
                    name=arg_name,
                    kind=ann_arg['kind'],
                    value=v
                ))
        elif ann_arg['kind'] == inspect.Parameter.VAR_KEYWORD:
            for var_arg_name, var_value in ann_arg['value'].items():
                flat_ann_args.append(dict(
                    name=var_arg_name,
                    kind=ann_arg['kind'],
                    value=var_value
                ))
        else:
            flat_ann_args.append(dict(
                name=arg_name,
                kind=ann_arg['kind'],
                value=ann_arg['value']
            ))
    return flat_ann_args


def match_ann_arg(ann_args: tp.AnnArgs, query: tp.AnnArgQuery) -> tp.Any:
    """Match an argument from annotated arguments.

    A query can be an integer indicating the position of the argument, or a string containing the name
    of the argument or a regular expression for matching the name of the argument.

    If multiple arguments were matched, returns the first one.

    The position can stretch over any variable argument."""
    flat_ann_args = flatten_ann_args(ann_args)
    if isinstance(query, int):
        return flat_ann_args[query]['value']
    if isinstance(query, str):
        for arg in flat_ann_args:
            if query == arg['name']:
                return arg['value']
        raise KeyError(f"Query '{query}' could not be matched with any argument")
    if isinstance(query, Regex):
        for arg in flat_ann_args:
            if query.matches(arg['name']):
                return arg['value']
        raise KeyError(f"Query '{query}' could not be matched with any argument")
    raise TypeError(f"Query of type {type(query)} is not supported")


def ignore_flat_ann_args(flat_ann_args: tp.FlatAnnArgs, ignore_args: tp.Iterable[tp.AnnArgQuery]) -> tp.FlatAnnArgs:
    """Ignore flattened annotated arguments."""
    new_flat_ann_args = []
    for i, arg in enumerate(flat_ann_args):
        arg_matched = False
        for ignore_arg in ignore_args:
            if isinstance(ignore_arg, int) and ignore_arg == i:
                arg_matched = True
                break
            if isinstance(ignore_arg, str) and ignore_arg == arg['name']:
                arg_matched = True
                break
            if isinstance(ignore_arg, Regex) and ignore_arg.matches(arg['name']):
                arg_matched = True
                break
        if not arg_matched:
            new_flat_ann_args.append(arg)
    return new_flat_ann_args


class UnhashableArgsError(Exception):
    """Unhashable arguments error."""
    pass


def hash_args(func: tp.Callable, args: tp.Args, kwargs: tp.Kwargs,
              ignore_args: tp.Optional[tp.Iterable[tp.AnnArgQuery]] = None) -> int:
    """Get hash of arguments.

    Use `ignore_args` to provide a sequence of queries for arguments that should be ignored."""
    if ignore_args is None:
        ignore_args = []
    ann_args = annotate_args(func, args, kwargs, only_passed=True)
    flat_ann_args = flatten_ann_args(ann_args)
    if len(ignore_args) > 0:
        flat_ann_args = ignore_flat_ann_args(flat_ann_args, ignore_args)
    try:
        return hash(tuple(map(lambda x: (x['name'], x['value']), flat_ann_args)))
    except TypeError:
        raise UnhashableArgsError


def get_ex_var_names(expression: str) -> tp.List[str]:
    """Get variable names listed in the expression."""
    return [node.id for node in ast.walk(ast.parse(expression)) if type(node) is ast.Name]


def get_context_vars(var_names: tp.Iterable[str],
                     frames_back: int = 0,
                     local_dict: tp.Optional[tp.Mapping] = None,
                     global_dict: tp.Optional[tp.Mapping] = None) -> tp.List[tp.Any]:
    """Get variables from the local/global context."""
    call_frame = sys._getframe(frames_back + 1)
    clear_local_dict = False
    if local_dict is None:
        local_dict = call_frame.f_locals
        clear_local_dict = True
    try:
        frame_globals = call_frame.f_globals
        if global_dict is None:
            global_dict = frame_globals
        clear_local_dict = clear_local_dict and frame_globals is not local_dict
        args = []
        for var_name in var_names:
            try:
                a = local_dict[var_name]
            except KeyError:
                a = global_dict[var_name]
            args.append(a)
    finally:
        # See https://github.com/pydata/numexpr/issues/310
        if clear_local_dict:
            local_dict.clear()
    return args
