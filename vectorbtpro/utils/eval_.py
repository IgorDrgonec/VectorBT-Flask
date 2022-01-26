# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Utilities for evaluation and compilation."""

import ast

from vectorbtpro import _typing as tp


def multiline_eval(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate several lines of input, returning the result of the last line."""
    if context is None:
        context = {}
    tree = ast.parse(expr)
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
    exec(compile(exec_expr, 'file', 'exec'), context)
    return eval(compile(eval_expr, 'file', 'eval'), context)
