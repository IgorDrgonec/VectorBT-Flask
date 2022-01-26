# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Factory for building indicators."""

import re
import inspect
import itertools
import functools
import warnings
import importlib
from collections import Counter, OrderedDict
from datetime import datetime, timedelta
from types import ModuleType

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List

from vectorbtpro import _typing as tp
from vectorbtpro.base import indexes, reshaping, combining
from vectorbtpro.base.indexing import build_param_indexer
from vectorbtpro.base.reshaping import Default, resolve_ref
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import BaseAccessor
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts, resolve_dict, Config, Configured
from vectorbtpro.utils.decorators import classproperty, cacheable_property, class_or_instancemethod
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.formatting import prettify
from vectorbtpro.utils.mapping import to_mapping, apply_mapping
from vectorbtpro.utils.params import to_typed_list, broadcast_params, create_param_product, params_to_list
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import has_templates, deep_substitute
from vectorbtpro.utils.parsing import get_expr_var_names, get_func_arg_names
from vectorbtpro.utils.eval_ import multiline_eval
from vectorbtpro.indicators.expr import expr_func_config, expr_res_func_config, wqa101_expr_config

try:
    from ta.utils import IndicatorMixin as IndicatorMixinT
except ImportError:
    IndicatorMixinT = tp.Any


def prepare_params(param_list: tp.Sequence[tp.Params],
                   param_names: tp.Sequence[str],
                   param_settings: tp.Sequence[tp.KwargsLike],
                   input_shape: tp.Optional[tp.Shape] = None,
                   to_2d: bool = False) -> tp.List[tp.Params]:
    """Prepare parameters.

    Resolves references and performs broadcasting to the input shape."""
    # Resolve references
    pool = dict(zip(param_names, param_list))
    for k in pool:
        pool[k] = resolve_ref(pool, k)
    param_list = [pool[k] for k in param_names]

    new_param_list = []
    for i, p_values in enumerate(param_list):
        # Resolve settings
        _param_settings = resolve_dict(param_settings[i])
        is_tuple = _param_settings.get('is_tuple', False)
        dtype = _param_settings.get('dtype', None)
        if checks.is_mapping_like(dtype):
            if checks.is_namedtuple(dtype):
                p_values = map_enum_fields(p_values, dtype)
            else:
                p_values = apply_mapping(p_values, dtype)
        is_array_like = _param_settings.get('is_array_like', False)
        bc_to_input = _param_settings.get('bc_to_input', False)
        broadcast_kwargs = merge_dicts(
            dict(require_kwargs=dict(requirements='W')),
            _param_settings.get('broadcast_kwargs', None)
        )

        new_p_values = params_to_list(p_values, is_tuple, is_array_like)
        if bc_to_input is not False:
            # Broadcast to input or its axis
            if is_tuple:
                raise ValueError("Cannot broadcast to input if tuple")
            if input_shape is None:
                raise ValueError("Cannot broadcast to input if input shape is unknown. Pass input_shape.")
            if bc_to_input is True:
                to_shape = input_shape
            else:
                checks.assert_in(bc_to_input, (0, 1))
                # Note that input_shape can be 1D
                if bc_to_input == 0:
                    to_shape = (input_shape[0],)
                else:
                    to_shape = (input_shape[1],) if len(input_shape) > 1 else (1,)
            _new_p_values = reshaping.broadcast(
                *new_p_values,
                to_shape=to_shape,
                **broadcast_kwargs
            )
            if len(new_p_values) == 1:
                _new_p_values = [_new_p_values]
            else:
                _new_p_values = list(_new_p_values)
            if to_2d and bc_to_input is True:
                # If inputs are meant to reshape to 2D, do the same to parameters
                # But only to those that fully resemble inputs (= not raw)
                __new_p_values = _new_p_values.copy()
                for j, param in enumerate(__new_p_values):
                    keep_flex = broadcast_kwargs.get('keep_flex', False)
                    if keep_flex is False or (isinstance(keep_flex, (tuple, list)) and not keep_flex[j]):
                        __new_p_values[j] = reshaping.to_2d(param)
                new_p_values = __new_p_values
            else:
                new_p_values = _new_p_values
        new_param_list.append(new_p_values)
    return new_param_list


def build_columns(param_list: tp.Sequence[tp.Params],
                  input_columns: tp.IndexLike,
                  level_names: tp.Optional[tp.Sequence[str]] = None,
                  hide_levels: tp.Optional[tp.Sequence[tp.Union[str, int]]] = None,
                  param_settings: tp.KwargsLikeSequence = None,
                  per_column: bool = False,
                  ignore_ranges: bool = False,
                  **kwargs) -> tp.Tuple[tp.List[tp.Index], tp.Index]:
    """For each parameter in `param_list`, create a new column level with parameter values
    and stack it on top of `input_columns`.

    Returns a list of parameter indexes and new columns."""
    if level_names is not None:
        checks.assert_len_equal(param_list, level_names)
    if hide_levels is None:
        hide_levels = []
    input_columns = indexes.to_any_index(input_columns)

    param_indexes = []
    shown_param_indexes = []
    for i in range(len(param_list)):
        p_values = param_list[i]
        level_name = None
        if level_names is not None:
            level_name = level_names[i]
        if per_column:
            param_index = indexes.index_from_values(p_values, name=level_name)
        else:
            _param_settings = resolve_dict(param_settings, i=i)
            _per_column = _param_settings.get('per_column', False)
            if _per_column:
                param_index = None
                for p in p_values:
                    bc_param = np.broadcast_to(p, (len(input_columns),))
                    _param_index = indexes.index_from_values(bc_param, name=level_name)
                    if param_index is None:
                        param_index = _param_index
                    else:
                        param_index = param_index.append(_param_index)
                if len(param_index) == 1 and len(input_columns) > 1:
                    # When using flexible column-wise parameters
                    param_index = indexes.repeat_index(
                        param_index,
                        len(input_columns),
                        ignore_ranges=ignore_ranges
                    )
            else:
                param_index = indexes.index_from_values(param_list[i], name=level_name)
                param_index = indexes.repeat_index(
                    param_index,
                    len(input_columns),
                    ignore_ranges=ignore_ranges
                )
        param_indexes.append(param_index)
        if i not in hide_levels and (level_names is None or level_names[i] not in hide_levels):
            shown_param_indexes.append(param_index)
    if not per_column:
        n_param_values = len(param_list[0]) if len(param_list) > 0 else 1
        input_columns = indexes.tile_index(
            input_columns,
            n_param_values,
            ignore_ranges=ignore_ranges
        )
    if len(shown_param_indexes) > 0:
        stacked_columns = indexes.stack_indexes([*shown_param_indexes, input_columns], **kwargs)
    else:
        stacked_columns = input_columns
    return param_indexes, stacked_columns


CacheOutputT = tp.Any
RawOutputT = tp.Tuple[tp.List[tp.Array2d], tp.List[tp.Tuple[tp.Param, ...]], int, tp.List[tp.Any]]
InputListT = tp.List[tp.Array2d]
InputMapperT = tp.Optional[tp.Array1d]
InOutputListT = tp.List[tp.Array2d]
OutputListT = tp.List[tp.Array2d]
ParamListT = tp.List[tp.List[tp.Param]]
MapperListT = tp.List[tp.Index]
OtherListT = tp.List[tp.Any]
PipelineOutputT = tp.Tuple[
    ArrayWrapper,
    InputListT,
    InputMapperT,
    InOutputListT,
    OutputListT,
    ParamListT,
    MapperListT,
    OtherListT
]


def run_pipeline(
        num_ret_outputs: int,
        custom_func: tp.Callable,
        *args,
        require_input_shape: bool = False,
        input_shape: tp.Optional[tp.ShapeLike] = None,
        input_index: tp.Optional[tp.IndexLike] = None,
        input_columns: tp.Optional[tp.IndexLike] = None,
        inputs: tp.Optional[tp.MappingSequence[tp.ArrayLike]] = None,
        in_outputs: tp.Optional[tp.MappingSequence[tp.ArrayLike]] = None,
        in_output_settings: tp.Optional[tp.MappingSequence[tp.KwargsLike]] = None,
        broadcast_named_args: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.Optional[tp.Mapping] = None,
        params: tp.Optional[tp.MappingSequence[tp.Params]] = None,
        param_product: bool = False,
        param_settings: tp.Optional[tp.MappingSequence[tp.KwargsLike]] = None,
        run_unique: bool = False,
        silence_warnings: bool = False,
        per_column: bool = False,
        pass_col: bool = False,
        keep_pd: bool = False,
        to_2d: bool = True,
        pass_packed: bool = False,
        pass_input_shape: tp.Optional[bool] = None,
        pass_flex_2d: bool = False,
        pass_wrapper: bool = False,
        level_names: tp.Optional[tp.Sequence[str]] = None,
        hide_levels: tp.Optional[tp.Sequence[tp.Union[str, int]]] = None,
        build_col_kwargs: tp.KwargsLike = None,
        return_raw: bool = False,
        use_raw: tp.Optional[RawOutputT] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        seed: tp.Optional[int] = None,
        **kwargs) -> tp.Union[CacheOutputT, RawOutputT, PipelineOutputT]:
    """A pipeline for running an indicator, used by `IndicatorFactory`.

    Args:
        num_ret_outputs (int): The number of output arrays returned by `custom_func`.
        custom_func (callable): A custom calculation function.

            See `IndicatorFactory.from_custom_func`.
        *args: Arguments passed to the `custom_func`.
        require_input_shape (bool): Whether to input shape is required.

            Will set `pass_input_shape` to True and raise an error if `input_shape` is None.
        input_shape (tuple): Shape to broadcast each input to.

            Can be passed to `custom_func`. See `pass_input_shape`.
        input_index (index_like): Sets index of each input.

            Can be used to label index if no inputs passed.
        input_columns (index_like): Sets columns of each input.

            Can be used to label columns if no inputs passed.
        inputs (mapping or sequence of array_like): A mapping or sequence of input arrays.

            Use mapping to also supply names. If sequence, will convert to a mapping using `input_{i}` key.
        in_outputs (mapping or sequence of array_like): A mapping or sequence of in-place output arrays.

            Use mapping to also supply names. If sequence, will convert to a mapping using `in_output_{i}` key.
        in_output_settings (dict or sequence of dict): Settings corresponding to each in-place output.

            If mapping, should contain keys from `in_outputs`.

            Following keys are accepted:

            * `dtype`: Create this array using this data type and `np.empty`. Default is None.
        broadcast_named_args (dict): Dictionary with named arguments to broadcast together with inputs.

            You can then pass argument names wrapped with `vectorbtpro.utils.template.Rep`
            and this method will substitute them by their corresponding broadcasted objects.
        broadcast_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`
            to broadcast inputs.
        template_context (dict): Mapping used to substitute templates in `args` and `kwargs`.
        params (mapping or sequence of any): A mapping or sequence of parameters.

            Use mapping to also supply names. If sequence, will convert to a mapping using `param_{i}` key.

            Each element is either an array-like object or a single value of any type.
        param_product (bool): Whether to build a Cartesian product out of all parameters.
        param_settings (dict or sequence of dict): Settings corresponding to each parameter.

            If mapping, should contain keys from `params`.

            Following keys are accepted:

            * `dtype`: If data type is an enumerated type or other mapping, and a string as parameter
                value was passed, will convert it first.
            * `is_tuple`: If tuple was passed, it will be considered as a single value.
                To treat it as multiple values, pack it into a list.
            * `is_array_like`: If array-like object was passed, it will be considered as a single value.
                To treat it as multiple values, pack it into a list.
            * `bc_to_input`: Whether to broadcast parameter to input size. You can also broadcast
                parameter to an axis by passing an integer.
            * `broadcast_kwargs`: Keyword arguments passed to `vectorbtpro.base.reshaping.broadcast`.
            * `per_column`: Whether each parameter value can be split per column such that it can
                be better reflected in a multi-index. Does not affect broadcasting.
        run_unique (bool): Whether to run only on unique parameter combinations.

            Disable if two identical parameter combinations can lead to different results
            (e.g., due to randomness) or if inputs are large and `custom_func` is fast.

            !!! note
                Cache, raw output, and output objects outside of `num_ret_outputs` will be returned
                for unique parameter combinations only.
        silence_warnings (bool): Whether to hide warnings such as coming from `run_unique`.
        per_column (bool): Whether to split the DataFrame into Series, one per column, and run `custom_func`
            on each Series.

            Each list of parameter values will broadcast to the number of columns and
            each parameter value will be applied per Series rather than per DataFrame.
            Input shape must be known beforehand.
        pass_col (bool): Whether to pass column index as keyword argument if `per_column` is set to True.
        keep_pd (bool): Whether to keep inputs as pandas objects, otherwise convert to NumPy arrays.
        to_2d (bool): Whether to reshape inputs to 2-dim arrays, otherwise keep as-is.
        pass_packed (bool): Whether to pass inputs and parameters to `custom_func` as lists.

            If `custom_func` is Numba-compiled, passes tuples.
        pass_input_shape (bool): Whether to pass `input_shape` to `custom_func` as keyword argument.

            Defaults to True if `require_input_shape` is True, otherwise to False.
        pass_flex_2d (bool): Whether to pass `flex_2d` to `custom_func` as keyword argument.
        pass_wrapper (bool): Whether to pass the input wrapper to `custom_func` as keyword argument.
        level_names (list of str): A list of column level names corresponding to each parameter.

            Must have the same length as `param_list`.
        hide_levels (list of int or str): A list of level names or indices of parameter levels to hide.
        build_col_kwargs (dict): Keyword arguments passed to `build_columns`.
        return_raw (bool): Whether to return raw output without post-processing and hashed parameter tuples.
        use_raw (bool): Takes the raw results and uses them instead of running `custom_func`.
        wrapper_kwargs (dict): Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper`.
        seed (int): Set seed to make output deterministic.
        **kwargs: Keyword arguments passed to the `custom_func`.

            Some common arguments include `return_cache` to return cache and `use_cache` to use cache.
            Those are only applicable to `custom_func` that supports it (`custom_func` created using
            `IndicatorFactory.from_apply_func` are supported by default).

    Returns:
        Array wrapper, list of inputs (`np.ndarray`), input mapper (`np.ndarray`), list of outputs
        (`np.ndarray`), list of parameter arrays (`np.ndarray`), list of parameter mappers (`np.ndarray`),
        list of outputs that are outside of `num_ret_outputs`.
    """
    if require_input_shape:
        checks.assert_not_none(input_shape)
        if pass_input_shape is None:
            pass_input_shape = True
    if pass_input_shape is None:
        pass_input_shape = False
    if input_index is not None:
        input_index = indexes.to_any_index(input_index)
    if input_columns is not None:
        input_columns = indexes.to_any_index(input_columns)
    if inputs is None:
        inputs = {}
    if not checks.is_mapping(inputs):
        inputs = {'input_' + str(i): input for i, input in enumerate(inputs)}
    input_names = list(inputs.keys())
    input_list = list(inputs.values())
    if in_outputs is None:
        in_outputs = {}
    if not checks.is_mapping(in_outputs):
        in_outputs = {'in_output_' + str(i): in_output for i, in_output in enumerate(in_outputs)}
    in_output_names = list(in_outputs.keys())
    in_output_list = list(in_outputs.values())
    if in_output_settings is None:
        in_output_settings = {}
    if checks.is_mapping(in_output_settings):
        checks.assert_dict_valid(in_output_settings, [in_output_names, 'dtype'])
        in_output_settings = [in_output_settings.get(k, None) for k in in_output_names]
    if broadcast_named_args is None:
        broadcast_named_args = {}
    if broadcast_kwargs is None:
        broadcast_kwargs = {}
    if template_context is None:
        template_context = {}
    if params is None:
        params = {}
    if not checks.is_mapping(params):
        params = {'param_' + str(i): param for i, param in enumerate(params)}
    param_names = list(params.keys())
    param_list = list(params.values())
    if param_settings is None:
        param_settings = {}
    if checks.is_mapping(param_settings):
        checks.assert_dict_valid(param_settings, [param_names, [
            'dtype',
            'is_tuple',
            'is_array_like',
            'bc_to_input',
            'broadcast_kwargs',
            'per_column'
        ]])
        param_settings = [param_settings.get(k, None) for k in param_names]
    if hide_levels is None:
        hide_levels = []
    if build_col_kwargs is None:
        build_col_kwargs = {}
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    if keep_pd and checks.is_numba_func(custom_func):
        raise ValueError("Cannot pass pandas objects to a Numba-compiled custom_func. Set keep_pd to False.")

    if input_shape is not None:
        input_shape = reshaping.shape_to_tuple(input_shape)
    if len(inputs) > 0 or len(in_outputs) > 0 or len(broadcast_named_args) > 0:
        # Broadcast inputs, in-outputs, and named args
        # If input_shape is provided, will broadcast all inputs to this shape
        broadcast_args = merge_dicts(inputs, in_outputs, broadcast_named_args)
        broadcast_kwargs = merge_dicts(dict(
            to_shape=input_shape,
            index_from=input_index,
            columns_from=input_columns,
            require_kwargs=dict(requirements='W'),
            post_func=np.asarray,
            to_pd=True
        ), broadcast_kwargs)
        broadcast_args, wrapper = reshaping.broadcast(
            broadcast_args,
            return_wrapper=True,
            **broadcast_kwargs
        )
        input_shape, input_index, input_columns = wrapper.shape, wrapper.index, wrapper.columns
        if input_index is None:
            input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
        if input_columns is None:
            input_columns = pd.RangeIndex(start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1)
        input_list = [broadcast_args[input_name] for input_name in input_names]
        in_output_list = [broadcast_args[in_output_name] for in_output_name in in_output_names]
        broadcast_named_args = {arg_name: broadcast_args[arg_name] for arg_name in broadcast_named_args}

    # Reshape input shape
    # Keep original input_shape for per_column=True
    orig_input_shape = input_shape
    orig_input_shape_2d = input_shape
    if input_shape is not None:
        orig_input_shape_2d = input_shape if len(input_shape) > 1 else (input_shape[0], 1)
    if per_column:
        # input_shape is now the size of one column
        if input_shape is None:
            raise ValueError("input_shape is required when per_column=True")
        input_shape = (input_shape[0],)
    input_shape_ready = input_shape
    input_shape_2d = input_shape
    if input_shape is not None:
        input_shape_2d = input_shape if len(input_shape) > 1 else (input_shape[0], 1)
    if to_2d:
        if input_shape is not None:
            input_shape_ready = input_shape_2d  # ready for custom_func

    # Prepare parameters
    # NOTE: input_shape instead of input_shape_ready since parameters should
    # broadcast by the same rules as inputs
    param_list = prepare_params(
        param_list,
        param_names,
        param_settings,
        input_shape=input_shape,
        to_2d=to_2d
    )
    if len(param_list) > 1:
        if level_names is not None:
            # Check level names
            checks.assert_len_equal(param_list, level_names)
            # Columns should be free of the specified level names
            if input_columns is not None:
                for level_name in level_names:
                    if level_name is not None:
                        checks.assert_level_not_exists(input_columns, level_name)
        if param_product:
            # Make Cartesian product out of all params
            param_list = create_param_product(param_list)
    if len(param_list) > 0:
        # Broadcast such that each array has the same length
        if per_column:
            # The number of parameters should match the number of columns before split
            param_list = broadcast_params(param_list, to_n=orig_input_shape_2d[1])
        else:
            param_list = broadcast_params(param_list)
    n_param_values = len(param_list[0]) if len(param_list) > 0 else 1
    use_run_unique = False
    param_list_unique = param_list
    if not per_column and run_unique:
        try:
            # Try to get all unique parameter combinations
            param_tuples = list(zip(*param_list))
            unique_param_tuples = list(OrderedDict.fromkeys(param_tuples).keys())
            if len(unique_param_tuples) < len(param_tuples):
                param_list_unique = list(map(list, zip(*unique_param_tuples)))
                use_run_unique = True
        except:
            pass
    if checks.is_numba_func(custom_func):
        # Numba can't stand untyped lists
        param_list_ready = [to_typed_list(params) for params in param_list_unique]
    else:
        param_list_ready = param_list_unique
    n_unique_param_values = len(param_list_unique[0]) if len(param_list_unique) > 0 else 1

    # Prepare inputs
    if per_column:
        # Split each input into Series/1-dim arrays, one per column
        input_list_ready = []
        for input in input_list:
            input_2d = reshaping.to_2d(input)
            col_inputs = []
            for i in range(input_2d.shape[1]):
                if to_2d:
                    col_input = input_2d[:, [i]]
                else:
                    col_input = input_2d[:, i]
                if keep_pd:
                    # Keep as pandas object
                    col_input = ArrayWrapper(input_index, input_columns[[i]], col_input.ndim).wrap(col_input)
                col_inputs.append(col_input)
            input_list_ready.append(col_inputs)
    else:
        input_list_ready = []
        for input in input_list:
            new_input = input
            if to_2d:
                new_input = reshaping.to_2d(input)
            if keep_pd:
                # Keep as pandas object
                new_input = ArrayWrapper(input_index, input_columns, new_input.ndim).wrap(new_input)
            input_list_ready.append(new_input)

    # Prepare in-place outputs
    in_output_list_ready = []
    for i in range(len(in_output_list)):
        if input_shape_2d is None:
            raise ValueError("input_shape is required when using in-place outputs")
        if in_output_list[i] is not None:
            # This in-place output has been already broadcast with inputs
            in_output_wide = np.require(in_output_list[i], requirements='W')
            if not per_column:
                # One per parameter combination
                in_output_wide = reshaping.tile(in_output_wide, n_unique_param_values, axis=1)
        else:
            # This in-place output hasn't been provided, so create empty
            _in_output_settings = resolve_dict(in_output_settings[i])
            dtype = _in_output_settings.get('dtype', None)
            in_output_shape = (input_shape_2d[0], input_shape_2d[1] * n_unique_param_values)
            in_output_wide = np.empty(in_output_shape, dtype=dtype)
        in_output_list[i] = in_output_wide
        in_outputs = []
        # Split each in-place output into chunks, each of input shape, and append to a list
        for i in range(n_unique_param_values):
            in_output = in_output_wide[:, i * input_shape_2d[1]: (i + 1) * input_shape_2d[1]]
            if len(input_shape_ready) == 1:
                in_output = in_output[:, 0]
            if keep_pd:
                if per_column:
                    in_output = ArrayWrapper(input_index, input_columns[[i]], in_output.ndim).wrap(in_output)
                else:
                    in_output = ArrayWrapper(input_index, input_columns, in_output.ndim).wrap(in_output)
            in_outputs.append(in_output)
        in_output_list_ready.append(in_outputs)
    if checks.is_numba_func(custom_func):
        # Numba can't stand untyped lists
        in_output_list_ready = [to_typed_list(in_outputs) for in_outputs in in_output_list_ready]

    def _use_raw(_raw):
        # Use raw results of previous run to build outputs
        _output_list, _param_map, _n_input_cols, _other_list = _raw
        idxs = np.array([_param_map.index(param_tuple) for param_tuple in zip(*param_list)])
        _output_list = [
            np.hstack([o[:, idx * _n_input_cols:(idx + 1) * _n_input_cols] for idx in idxs])
            for o in _output_list
        ]
        return _output_list, _param_map, _n_input_cols, _other_list

    # Get raw results
    if use_raw is not None:
        # Use raw results of previous run to build outputs
        output_list, param_map, n_input_cols, other_list = _use_raw(use_raw)
    else:
        # Prepare other arguments
        func_args = args
        func_kwargs = {}
        if pass_input_shape:
            func_kwargs['input_shape'] = input_shape_ready
        if pass_flex_2d:
            if input_shape is None:
                raise ValueError("Cannot determine flex_2d without inputs")
            func_kwargs['flex_2d'] = len(input_shape) == 2
        if pass_wrapper:
            func_kwargs['wrapper'] = ArrayWrapper(input_index, input_columns, len(orig_input_shape))
        func_kwargs = merge_dicts(func_kwargs, kwargs)

        # Set seed
        if seed is not None:
            set_seed(seed)

        def _call_custom_func(_input_list_ready, _in_output_list_ready, _param_list_ready, *_func_args, **_func_kwargs):
            # Substitute templates
            if has_templates(_func_args) or has_templates(_func_kwargs):
                _template_context = merge_dicts(
                    broadcast_named_args,
                    dict(
                        input_shape=input_shape_ready,
                        **dict(zip(input_names, _input_list_ready)),
                        **dict(zip(in_output_names, _in_output_list_ready)),
                        **dict(zip(param_names, _param_list_ready)),
                        pre_sub_args=_func_args,
                        pre_sub_kwargs=_func_kwargs
                    ),
                    template_context
                )
                _func_args = deep_substitute(_func_args, _template_context, sub_id='custom_func_args')
                _func_kwargs = deep_substitute(_func_kwargs, _template_context, sub_id='custom_func_kwargs')

            # Run the function
            if pass_packed:
                if checks.is_numba_func(custom_func):
                    return custom_func(
                        tuple(_input_list_ready),
                        tuple(_in_output_list_ready),
                        tuple(_param_list_ready),
                        *_func_args,
                        **_func_kwargs
                    )
                return custom_func(
                    _input_list_ready,
                    _in_output_list_ready,
                    _param_list_ready,
                    *_func_args,
                    **_func_kwargs
                )
            return custom_func(
                *_input_list_ready,
                *_in_output_list_ready,
                *_param_list_ready,
                *_func_args,
                **_func_kwargs
            )

        if per_column:
            output = []
            for col in range(orig_input_shape_2d[1]):
                # Select the column of each input and in-place output, and the respective parameter combination
                _input_list_ready = []
                for _inputs in input_list_ready:
                    # Each input array is now one column wide
                    _input_list_ready.append(_inputs[col])

                _in_output_list_ready = []
                for _in_outputs in in_output_list_ready:
                    # Each in-output array is now one column wide
                    if isinstance(_in_outputs, List):
                        __in_outputs = List()
                    else:
                        __in_outputs = []
                    __in_outputs.append(_in_outputs[col])
                    _in_output_list_ready.append(__in_outputs)

                _param_list_ready = []
                for _params in param_list_ready:
                    # Each parameter list is now one element long
                    if isinstance(_params, List):
                        __params = List()
                    else:
                        __params = []
                    __params.append(_params[col])
                    _param_list_ready.append(__params)

                _func_args = func_args
                _func_kwargs = func_kwargs.copy()
                if 'use_cache' in func_kwargs:
                    use_cache = func_kwargs['use_cache']
                    if isinstance(use_cache, list) and len(use_cache) == orig_input_shape_2d[1]:
                        # Pass cache for this column
                        _func_kwargs['use_cache'] = func_kwargs['use_cache'][col]
                if pass_col:
                    _func_kwargs['col'] = col
                col_output = _call_custom_func(
                    _input_list_ready,
                    _in_output_list_ready,
                    _param_list_ready,
                    *_func_args,
                    **_func_kwargs
                )
                output.append(col_output)
        else:
            output = _call_custom_func(
                input_list_ready,
                in_output_list_ready,
                param_list_ready,
                *func_args,
                **func_kwargs
            )

        # Return cache
        if kwargs.get('return_cache', False):
            if use_run_unique and not silence_warnings:
                warnings.warn("Cache is produced by unique parameter "
                              "combinations when run_unique=True", stacklevel=2)
            return output

        def _split_output(output):
            # Post-process results
            if output is None:
                _output_list = []
                _other_list = []
            else:
                if isinstance(output, (tuple, list, List)):
                    _output_list = list(output)
                else:
                    _output_list = [output]
                # Other outputs should be returned without post-processing (for example cache_dict)
                if len(_output_list) > num_ret_outputs:
                    _other_list = _output_list[num_ret_outputs:]
                    if use_run_unique and not silence_warnings:
                        warnings.warn("Additional output objects are produced by unique parameter "
                                      "combinations when run_unique=True", stacklevel=2)
                else:
                    _other_list = []
                # Process only the num_ret_outputs outputs
                _output_list = _output_list[:num_ret_outputs]
            if len(_output_list) != num_ret_outputs:
                raise ValueError("Number of returned outputs other than expected")
            _output_list = list(map(lambda x: reshaping.to_2d_array(x), _output_list))
            return _output_list, _other_list

        if per_column:
            output_list = []
            other_list = []
            for _output in output:
                __output_list, __other_list = _split_output(_output)
                output_list.append(__output_list)
                if len(__other_list) > 0:
                    other_list.append(__other_list)
            # Concatenate each output (must be one column wide)
            output_list = [np.hstack(input_group) for input_group in zip(*output_list)]
        else:
            output_list, other_list = _split_output(output)

        # In-place outputs are treated as outputs from here
        output_list = in_output_list + output_list

        # Prepare raw
        param_map = list(zip(*param_list_unique))  # account for use_run_unique
        output_shape = output_list[0].shape
        for output in output_list:
            if output.shape != output_shape:
                raise ValueError("All outputs must have the same shape")
        if per_column:
            n_input_cols = 1
        else:
            n_input_cols = output_shape[1] // n_unique_param_values
        if input_shape_2d is not None:
            if n_input_cols != input_shape_2d[1]:
                if per_column:
                    raise ValueError("All outputs must have one column when per_column=True")
                else:
                    raise ValueError("All outputs must have the number of columns = #input columns x #parameters")
        raw = output_list, param_map, n_input_cols, other_list
        if return_raw:
            if use_run_unique and not silence_warnings:
                warnings.warn("Raw output is produced by unique parameter "
                              "combinations when run_unique=True", stacklevel=2)
            return raw
        if use_run_unique:
            output_list, param_map, n_input_cols, other_list = _use_raw(raw)

    # Update shape and other meta if no inputs
    if input_shape is None:
        if n_input_cols == 1:
            input_shape = (output_list[0].shape[0],)
        else:
            input_shape = (output_list[0].shape[0], n_input_cols)
    else:
        input_shape = orig_input_shape
    if input_index is None:
        input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
    if input_columns is None:
        input_columns = pd.RangeIndex(start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1)

    # Build column hierarchy and create mappers
    if len(param_list) > 0:
        # Build new column levels on top of input levels
        param_indexes, new_columns = build_columns(
            param_list,
            input_columns,
            level_names=level_names,
            hide_levels=hide_levels,
            param_settings=param_settings,
            per_column=per_column,
            **build_col_kwargs
        )
        # Build a mapper that maps old columns in inputs to new columns
        # Instead of tiling all inputs to the shape of outputs and wasting memory,
        # we just keep a mapper and perform the tiling when needed
        input_mapper = None
        if len(input_list) > 0:
            if per_column:
                input_mapper = np.arange(len(input_columns))
            else:
                input_mapper = np.tile(np.arange(len(input_columns)), n_param_values)
        # Build mappers to easily map between parameters and columns
        mapper_list = [param_indexes[i] for i in range(len(param_list))]
    else:
        # Some indicators don't have any params
        new_columns = input_columns
        input_mapper = None
        mapper_list = []

    # Return artifacts: no pandas objects, just a wrapper and NumPy arrays
    new_ndim = len(input_shape) if output_list[0].shape[1] == 1 else output_list[0].ndim
    wrapper = ArrayWrapper(input_index, new_columns, new_ndim, **wrapper_kwargs)

    return wrapper, \
           input_list, \
           input_mapper, \
           output_list[:len(in_output_list)], \
           output_list[len(in_output_list):], \
           param_list, \
           mapper_list, \
           other_list


def combine_objs(obj: tp.SeriesFrame,
                 other: tp.MaybeTupleList[tp.Union[tp.ArrayLike, BaseAccessor]],
                 combine_func: tp.Callable,
                 *args, level_name: tp.Optional[str] = None,
                 keys: tp.Optional[tp.IndexLike] = None,
                 allow_multiple: bool = True,
                 **kwargs) -> tp.SeriesFrame:
    """Combines/compares `obj` to `other`, for example, to generate signals.

    Both will broadcast together.
    Pass `other` as a tuple or a list to compare with multiple arguments.
    In this case, a new column level will be created with the name `level_name`.

    See `vectorbtpro.base.accessors.BaseAccessor.combine`."""
    if allow_multiple and isinstance(other, (tuple, list)):
        if keys is None:
            keys = indexes.index_from_values(other, name=level_name)
    return obj.vbt.combine(other, combine_func, *args, keys=keys, allow_multiple=allow_multiple, **kwargs)


IndicatorBaseT = tp.TypeVar("IndicatorBaseT", bound="IndicatorBase")
RunOutputT = tp.Union[IndicatorBaseT, tp.Tuple[tp.Any, ...], RawOutputT, CacheOutputT]
RunCombsOutputT = tp.Tuple[IndicatorBaseT, ...]


class IndicatorBase(Analyzable):
    """Indicator base class.

    Properties should be set before instantiation."""

    _short_name: str
    _level_names: tp.Tuple[str, ...]
    _input_names: tp.Tuple[str, ...]
    _param_names: tp.Tuple[str, ...]
    _in_output_names: tp.Tuple[str, ...]
    _output_names: tp.Tuple[str, ...]
    _output_flags: tp.Kwargs

    @property
    def short_name(self) -> str:
        """Name of the indicator."""
        return self._short_name

    @property
    def level_names(self) -> tp.Tuple[str, ...]:
        """Column level names corresponding to each parameter."""
        return self._level_names

    @classproperty
    def input_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the input arrays."""
        return cls_or_self._input_names

    @classproperty
    def param_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the parameters."""
        return cls_or_self._param_names

    @classproperty
    def in_output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the in-place output arrays."""
        return cls_or_self._in_output_names

    @classproperty
    def output_names(cls_or_self) -> tp.Tuple[str, ...]:
        """Names of the regular output arrays."""
        return cls_or_self._output_names

    @classproperty
    def output_flags(cls_or_self) -> tp.Kwargs:
        """Dictionary of output flags."""
        return cls_or_self._output_flags

    def __init__(self,
                 wrapper: ArrayWrapper,
                 input_list: InputListT,
                 input_mapper: InputMapperT,
                 in_output_list: InOutputListT,
                 output_list: OutputListT,
                 param_list: ParamListT,
                 mapper_list: MapperListT,
                 short_name: str,
                 level_names: tp.Tuple[str, ...],
                 **kwargs) -> None:
        if input_mapper is not None:
            checks.assert_equal(input_mapper.shape[0], wrapper.shape_2d[1])
        for ts in input_list:
            checks.assert_equal(ts.shape[0], wrapper.shape_2d[0])
        for ts in in_output_list + output_list:
            checks.assert_equal(ts.shape, wrapper.shape_2d)
        for params in param_list:
            checks.assert_len_equal(param_list[0], params)
        for mapper in mapper_list:
            checks.assert_equal(len(mapper), wrapper.shape_2d[1])
        checks.assert_instance_of(short_name, str)
        checks.assert_len_equal(level_names, param_list)

        Analyzable.__init__(
            self,
            wrapper,
            input_list=input_list,
            input_mapper=input_mapper,
            in_output_list=in_output_list,
            output_list=output_list,
            param_list=param_list,
            mapper_list=mapper_list,
            short_name=short_name,
            level_names=level_names,
            **kwargs
        )

        setattr(self, '_short_name', short_name)
        setattr(self, '_level_names', level_names)

        for i, ts_name in enumerate(self.input_names):
            setattr(self, f'_{ts_name}', input_list[i])
        setattr(self, '_input_mapper', input_mapper)
        for i, in_output_name in enumerate(self.in_output_names):
            setattr(self, f'_{in_output_name}', in_output_list[i])
        for i, output_name in enumerate(self.output_names):
            setattr(self, f'_{output_name}', output_list[i])
        for i, param_name in enumerate(self.param_names):
            setattr(self, f'_{param_name}_list', param_list[i])
            setattr(self, f'_{param_name}_mapper', mapper_list[i])
        if len(self.param_names) > 1:
            tuple_mapper = list(zip(*list(mapper_list)))
            setattr(self, '_tuple_mapper', tuple_mapper)

    def indexing_func(self: IndicatorBaseT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> IndicatorBaseT:
        """Perform indexing on `IndicatorBase`."""
        new_wrapper, idx_idxs, _, col_idxs = self.wrapper.indexing_func_meta(pd_indexing_func, **kwargs)
        idx_idxs_arr = reshaping.to_1d_array(idx_idxs)
        col_idxs_arr = reshaping.to_1d_array(col_idxs)
        if np.array_equal(idx_idxs_arr, np.arange(self.wrapper.shape_2d[0])):
            idx_idxs_arr = slice(None, None, None)
        if np.array_equal(col_idxs_arr, np.arange(self.wrapper.shape_2d[1])):
            col_idxs_arr = slice(None, None, None)

        input_mapper = getattr(self, '_input_mapper', None)
        if input_mapper is not None:
            input_mapper = input_mapper[col_idxs_arr]
        input_list = []
        for input_name in self.input_names:
            input_list.append(getattr(self, f'_{input_name}')[idx_idxs_arr])
        in_output_list = []
        for in_output_name in self.in_output_names:
            in_output_list.append(getattr(self, f'_{in_output_name}')[idx_idxs_arr, :][:, col_idxs_arr])
        output_list = []
        for output_name in self.output_names:
            output_list.append(getattr(self, f'_{output_name}')[idx_idxs_arr, :][:, col_idxs_arr])
        param_list = []
        for param_name in self.param_names:
            param_list.append(getattr(self, f'_{param_name}_list'))
        mapper_list = []
        for param_name in self.param_names:
            # Tuple mapper is a list because of its complex data type
            mapper_list.append(getattr(self, f'_{param_name}_mapper')[col_idxs_arr])

        return self.replace(
            wrapper=new_wrapper,
            input_list=input_list,
            input_mapper=input_mapper,
            in_output_list=in_output_list,
            output_list=output_list,
            param_list=param_list,
            mapper_list=mapper_list
        )

    @classmethod
    def _run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
        """Private run method."""
        raise NotImplementedError

    @classmethod
    def run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
        """Public run method."""
        return cls._run(*args, **kwargs)

    @classmethod
    def _run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
        """Private run combinations method."""
        raise NotImplementedError

    @classmethod
    def run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
        """Public run combinations method."""
        return cls._run_combs(*args, **kwargs)


class IndicatorFactory(Configured):
    def __init__(self,
                 class_name: str = 'Indicator',
                 class_docstring: str = '',
                 module_name: tp.Optional[str] = __name__,
                 short_name: tp.Optional[str] = None,
                 prepend_name: bool = True,
                 input_names: tp.Optional[tp.Sequence[str]] = None,
                 param_names: tp.Optional[tp.Sequence[str]] = None,
                 in_output_names: tp.Optional[tp.Sequence[str]] = None,
                 output_names: tp.Optional[tp.Sequence[str]] = None,
                 output_flags: tp.KwargsLike = None,
                 custom_output_props: tp.KwargsLike = None,
                 attr_settings: tp.KwargsLike = None,
                 metrics: tp.Optional[tp.Kwargs] = None,
                 stats_defaults: tp.Union[None, tp.Callable, tp.Kwargs] = None,
                 subplots: tp.Optional[tp.Kwargs] = None,
                 plots_defaults: tp.Union[None, tp.Callable, tp.Kwargs] = None) -> None:
        """A factory for creating new indicators.

        Initialize `IndicatorFactory` to create a skeleton and then use a class method
        such as `IndicatorFactory.from_custom_func` to bind a calculation function to the skeleton.

        Args:
            class_name (str): Name for the created indicator class.
            class_docstring (str): Docstring for the created indicator class.
            module_name (str): Name of the module the class originates from.
            short_name (str): Short name of the indicator.

                Defaults to lower-case `class_name`.
            prepend_name (bool): Whether to prepend `short_name` to each parameter level.
            input_names (list of str): List with input names.
            param_names (list of str): List with parameter names.
            in_output_names (list of str): List with in-output names.

                An in-place output is an output that is not returned but modified in-place.
                Some advantages of such outputs include:

                1) they don't need to be returned,
                2) they can be passed between functions as easily as inputs,
                3) they can be provided with already allocated data to safe memory,
                4) if data or default value are not provided, they are created empty to not occupy memory.
            output_names (list of str): List with output names.
            output_flags (dict): Dictionary of in-place and regular output flags.
            custom_output_props (dict): Dictionary with user-defined functions that will be
                bound to the indicator class and wrapped with `vectorbtpro.utils.decorators.cacheable_property`.
            attr_settings (dict): Dictionary with attribute settings.

                Attributes can be `input_names`, `in_output_names`, `output_names` and `custom_output_props`.

                Following keys are accepted:

                * `dtype`: Data type used to determine which methods to generate around this attribute.
                    Set to None to disable. Default is `np.float_`. Can be set to instance of
                    `collections.namedtuple` acting as enumerated type, or any other mapping;
                    It will then create a property with suffix `readable` that contains data in a string format.
            metrics (dict): Metrics supported by `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

                If dict, will be converted to `vectorbtpro.utils.config.Config`.
            stats_defaults (callable or dict): Defaults for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`.

                If dict, will be converted into a property.
            subplots (dict): Subplots supported by `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

                If dict, will be converted to `vectorbtpro.utils.config.Config`.
            plots_defaults (callable or dict): Defaults for `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`.

                If dict, will be converted into a property.

        !!! note
            The `__init__` method is not used for running the indicator, for this use `run`.
            The reason for this is indexing, which requires a clean `__init__` method for creating
            a new indicator object with newly indexed attributes.
        """
        Configured.__init__(
            self,
            class_name=class_name,
            class_docstring=class_docstring,
            module_name=module_name,
            short_name=short_name,
            prepend_name=prepend_name,
            input_names=input_names,
            param_names=param_names,
            in_output_names=in_output_names,
            output_names=output_names,
            output_flags=output_flags,
            custom_output_props=custom_output_props,
            attr_settings=attr_settings,
            metrics=metrics,
            stats_defaults=stats_defaults,
            subplots=subplots,
            plots_defaults=plots_defaults
        )

        # Check parameters
        checks.assert_instance_of(class_name, str)
        checks.assert_instance_of(class_docstring, str)
        if module_name is not None:
            checks.assert_instance_of(module_name, str)
        if short_name is None:
            if class_name == 'Indicator':
                short_name = 'custom'
            else:
                short_name = class_name.lower()
        checks.assert_instance_of(short_name, str)
        checks.assert_instance_of(prepend_name, bool)
        if input_names is None:
            input_names = []
        else:
            checks.assert_sequence(input_names)
            input_names = list(input_names)
        if param_names is None:
            param_names = []
        else:
            checks.assert_sequence(param_names)
            param_names = list(param_names)
        if in_output_names is None:
            in_output_names = []
        else:
            checks.assert_sequence(in_output_names)
            in_output_names = list(in_output_names)
        if output_names is None:
            output_names = []
        else:
            checks.assert_sequence(output_names)
            output_names = list(output_names)
        all_output_names = in_output_names + output_names
        if len(all_output_names) == 0:
            raise ValueError("Must have at least one in-place or regular output")
        if len(set.intersection(set(input_names), set(in_output_names), set(output_names))) > 0:
            raise ValueError("Inputs, in-outputs, and parameters must all have unique names")
        if output_flags is None:
            output_flags = {}
        checks.assert_instance_of(output_flags, dict)
        if len(output_flags) > 0:
            checks.assert_dict_valid(output_flags, all_output_names)
        if custom_output_props is None:
            custom_output_props = {}
        checks.assert_instance_of(custom_output_props, dict)
        if attr_settings is None:
            attr_settings = {}
        checks.assert_instance_of(attr_settings, dict)
        all_attr_names = input_names + all_output_names + list(custom_output_props.keys())
        if len(attr_settings) > 0:
            checks.assert_dict_valid(attr_settings, all_attr_names)

        # Set up class
        ParamIndexer = build_param_indexer(
            param_names + (['tuple'] if len(param_names) > 1 else []),
            module_name=module_name
        )
        Indicator = type(class_name, (IndicatorBase, ParamIndexer), {})
        Indicator.__doc__ = class_docstring
        if module_name is not None:
            Indicator.__module__ = module_name

        # Create read-only properties
        setattr(Indicator, "_input_names", tuple(input_names))
        setattr(Indicator, "_param_names", tuple(param_names))
        setattr(Indicator, "_in_output_names", tuple(in_output_names))
        setattr(Indicator, "_output_names", tuple(output_names))
        setattr(Indicator, "_output_flags", output_flags)

        for param_name in param_names:
            def param_list_prop(self, _param_name=param_name) -> tp.List[tp.Param]:
                return getattr(self, f'_{_param_name}_list')

            param_list_prop.__doc__ = f"List of `{param_name}` values."
            setattr(Indicator, f'{param_name}_list', property(param_list_prop))

        for input_name in input_names:
            def input_prop(self, _input_name: str = input_name) -> tp.SeriesFrame:
                """Input array."""
                old_input = reshaping.to_2d_array(getattr(self, '_' + _input_name))
                input_mapper = getattr(self, '_input_mapper')
                if input_mapper is None:
                    return self.wrapper.wrap(old_input)
                return self.wrapper.wrap(old_input[:, input_mapper])

            input_prop.__name__ = input_name
            setattr(Indicator, input_name, cacheable_property(input_prop))

        for output_name in all_output_names:
            def output_prop(self, _output_name: str = output_name) -> tp.SeriesFrame:
                return self.wrapper.wrap(getattr(self, '_' + _output_name))

            if output_name in in_output_names:
                output_prop.__doc__ = """In-place output array."""
            else:
                output_prop.__doc__ = """Output array."""

            output_prop.__name__ = output_name
            if output_name in output_flags:
                _output_flags = output_flags[output_name]
                if isinstance(_output_flags, (tuple, list)):
                    _output_flags = ', '.join(_output_flags)
                output_prop.__doc__ += "\n\n" + _output_flags
            setattr(Indicator, output_name, property(output_prop))

        # Add __init__ method
        def __init__(self,
                     wrapper: ArrayWrapper,
                     input_list: InputListT,
                     input_mapper: InputMapperT,
                     in_output_list: InOutputListT,
                     output_list: OutputListT,
                     param_list: ParamListT,
                     mapper_list: MapperListT,
                     short_name: str,
                     level_names: tp.Tuple[str, ...]) -> None:
            IndicatorBase.__init__(
                self,
                wrapper,
                input_list,
                input_mapper,
                in_output_list,
                output_list,
                param_list,
                mapper_list,
                short_name,
                level_names
            )
            if len(param_names) > 1:
                tuple_mapper = list(zip(*list(mapper_list)))
            else:
                tuple_mapper = None

            # Initialize indexers
            mapper_sr_list = []
            for i, m in enumerate(mapper_list):
                mapper_sr_list.append(pd.Series(m, index=wrapper.columns))
            if tuple_mapper is not None:
                mapper_sr_list.append(pd.Series(tuple_mapper, index=wrapper.columns))
            ParamIndexer.__init__(self, mapper_sr_list, level_names=[*level_names, level_names])

        setattr(Indicator, '__init__', __init__)

        # Add user-defined outputs
        for prop_name, prop in custom_output_props.items():
            if prop.__doc__ is None:
                prop.__doc__ = f"""Custom property."""
            prop.__name__ = prop_name
            prop = cacheable_property(prop)
            setattr(Indicator, prop_name, prop)

        # Add comparison & combination methods for all inputs, outputs, and user-defined properties
        def assign_combine_method(func_name: str,
                                  combine_func: tp.Callable,
                                  def_kwargs: tp.Kwargs,
                                  attr_name: str,
                                  docstring: str) -> None:
            def combine_method(self: IndicatorBaseT,
                               other: tp.MaybeTupleList[tp.Union[IndicatorBaseT, tp.ArrayLike, BaseAccessor]],
                               level_name: tp.Optional[str] = None,
                               allow_multiple: bool = True,
                               _prepend_name: bool = prepend_name,
                               **kwargs) -> tp.SeriesFrame:
                if allow_multiple and isinstance(other, (tuple, list)):
                    other = list(other)
                    for i in range(len(other)):
                        if isinstance(other[i], IndicatorBase):
                            other[i] = getattr(other[i], attr_name)
                else:
                    if isinstance(other, IndicatorBase):
                        other = getattr(other, attr_name)
                if level_name is None:
                    if _prepend_name:
                        if attr_name == self.short_name:
                            level_name = f'{self.short_name}_{func_name}'
                        else:
                            level_name = f'{self.short_name}_{attr_name}_{func_name}'
                    else:
                        level_name = f'{attr_name}_{func_name}'
                out = combine_objs(
                    getattr(self, attr_name),
                    other,
                    combine_func,
                    level_name=level_name,
                    allow_multiple=allow_multiple,
                    **merge_dicts(def_kwargs, kwargs)
                )
                return out

            combine_method.__qualname__ = f'{Indicator.__name__}.{attr_name}_{func_name}'
            combine_method.__doc__ = docstring
            setattr(Indicator, f'{attr_name}_{func_name}', combine_method)

        for attr_name in all_attr_names:
            _attr_settings = attr_settings.get(attr_name, {})
            checks.assert_dict_valid(_attr_settings, ['dtype'])
            dtype = _attr_settings.get('dtype', np.float_)

            if checks.is_mapping_like(dtype):
                def attr_readable(self,
                                  _attr_name: str = attr_name,
                                  _mapping: tp.MappingLike = dtype) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt(mapping=_mapping).apply_mapping()

                attr_readable.__qualname__ = f'{Indicator.__name__}.{attr_name}_readable'
                attr_readable.__doc__ = inspect.cleandoc(
                    """`{attr_name}` in readable format based on the following mapping: 
                                
                    ```python
                    {dtype}
                    ```"""
                ).format(
                    attr_name=attr_name,
                    dtype=prettify(to_mapping(dtype))
                )
                setattr(Indicator, f'{attr_name}_readable', property(attr_readable))

                def attr_stats(self, *args,
                               _attr_name: str = attr_name,
                               _mapping: tp.MappingLike = dtype,
                               **kwargs) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt(mapping=_mapping).stats(*args, **kwargs)

                attr_stats.__qualname__ = f'{Indicator.__name__}.{attr_name}_stats'
                attr_stats.__doc__ = inspect.cleandoc(
                    """Stats of `{attr_name}` based on the following mapping: 

                    ```python
                    {dtype}
                    ```"""
                ).format(
                    attr_name=attr_name,
                    dtype=prettify(to_mapping(dtype))
                )
                setattr(Indicator, f'{attr_name}_stats', attr_stats)

            elif np.issubdtype(dtype, np.number):
                func_info = [
                    ('above', np.greater, dict()),
                    ('below', np.less, dict()),
                    ('equal', np.equal, dict()),
                    ('crossed_above',
                     lambda x, y, wait=0:
                     jit_reg.resolve(generic_nb.crossed_above_nb)(x, y, wait),
                     dict(to_2d=True)),
                    ('crossed_below',
                     lambda x, y, wait=0:
                     jit_reg.resolve(generic_nb.crossed_above_nb)(y, x, wait),
                     dict(to_2d=True))
                ]
                for func_name, np_func, def_kwargs in func_info:
                    method_docstring = f"""Return True for each element where `{attr_name}` is {func_name} `other`. 
                
                    See `vectorbtpro.indicators.factory.combine_objs`."""
                    assign_combine_method(func_name, np_func, def_kwargs, attr_name, method_docstring)

                def attr_stats(self, *args, _attr_name: str = attr_name, **kwargs) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt.stats(*args, **kwargs)

                attr_stats.__qualname__ = f'{Indicator.__name__}.{attr_name}_stats'
                attr_stats.__doc__ = f"""Stats of `{attr_name}` as generic."""
                setattr(Indicator, f'{attr_name}_stats', attr_stats)

            elif np.issubdtype(dtype, np.bool_):
                func_info = [
                    ('and', np.logical_and, dict()),
                    ('or', np.logical_or, dict()),
                    ('xor', np.logical_xor, dict())
                ]
                for func_name, np_func, def_kwargs in func_info:
                    method_docstring = f"""Return `{attr_name} {func_name.upper()} other`. 

                    See `vectorbtpro.indicators.factory.combine_objs`."""
                    assign_combine_method(func_name, np_func, def_kwargs, attr_name, method_docstring)

                def attr_stats(self, *args, _attr_name: str = attr_name, **kwargs) -> tp.SeriesFrame:
                    return getattr(self, _attr_name).vbt.signals.stats(*args, **kwargs)

                attr_stats.__qualname__ = f'{Indicator.__name__}.{attr_name}_stats'
                attr_stats.__doc__ = f"""Stats of `{attr_name}` as signals."""
                setattr(Indicator, f'{attr_name}_stats', attr_stats)

        # Prepare stats
        if metrics is not None:
            if not isinstance(metrics, Config):
                metrics = Config(metrics, copy_kwargs_=dict(copy_mode='deep'))
            setattr(Indicator, "_metrics", metrics.copy())

        if stats_defaults is not None:
            if isinstance(stats_defaults, dict):
                def stats_defaults_prop(self, _stats_defaults: tp.Kwargs = stats_defaults) -> tp.Kwargs:
                    return _stats_defaults
            else:
                def stats_defaults_prop(self, _stats_defaults: tp.Kwargs = stats_defaults) -> tp.Kwargs:
                    return stats_defaults(self)
            stats_defaults_prop.__name__ = "stats_defaults"
            setattr(Indicator, "stats_defaults", property(stats_defaults_prop))

        # Prepare plots
        if subplots is not None:
            if not isinstance(subplots, Config):
                subplots = Config(subplots, copy_kwargs_=dict(copy_mode='deep'))
            setattr(Indicator, "_subplots", subplots.copy())

        if plots_defaults is not None:
            if isinstance(plots_defaults, dict):
                def plots_defaults_prop(self, _plots_defaults: tp.Kwargs = plots_defaults) -> tp.Kwargs:
                    return _plots_defaults
            else:
                def plots_defaults_prop(self, _plots_defaults: tp.Kwargs = plots_defaults) -> tp.Kwargs:
                    return plots_defaults(self)
            plots_defaults_prop.__name__ = "plots_defaults"
            setattr(Indicator, "plots_defaults", property(plots_defaults_prop))

        # Store arguments
        self._class_name = class_name
        self._class_docstring = class_docstring
        self._module_name = module_name
        self._short_name = short_name
        self._prepend_name = prepend_name
        self._input_names = input_names
        self._param_names = param_names
        self._in_output_names = in_output_names
        self._output_names = output_names
        self._output_flags = output_flags
        self._custom_output_props = custom_output_props
        self._attr_settings = attr_settings
        self._metrics = metrics
        self._stats_defaults = stats_defaults
        self._subplots = subplots
        self._plots_defaults = plots_defaults

        # Store indicator class
        self._Indicator = Indicator

    @property
    def class_name(self):
        """Name for the created indicator class."""
        return self._class_name

    @property
    def class_docstring(self):
        """Docstring for the created indicator class."""
        return self._class_docstring

    @property
    def module_name(self):
        """Name of the module the class originates from."""
        return self._module_name

    @property
    def short_name(self):
        """Short name of the indicator."""
        return self._short_name

    @property
    def prepend_name(self):
        """Whether to prepend `IndicatorFactory.short_name` to each parameter level."""
        return self._prepend_name

    @property
    def input_names(self):
        """List with input names."""
        return self._input_names

    @property
    def param_names(self):
        """List with parameter names."""
        return self._param_names

    @property
    def in_output_names(self):
        """List with in-output names."""
        return self._in_output_names

    @property
    def output_names(self):
        """List with output names."""
        return self._output_names

    @property
    def output_flags(self):
        """Dictionary of in-place and regular output flags."""
        return self._output_flags

    @property
    def custom_output_props(self):
        """Dictionary with user-defined functions that will become properties."""
        return self._custom_output_props

    @property
    def attr_settings(self):
        """Dictionary with attribute settings."""
        return self._attr_settings

    @property
    def metrics(self):
        """Metrics supported by `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`."""
        return self._metrics

    @property
    def stats_defaults(self):
        """Defaults for `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats`."""
        return self._stats_defaults

    @property
    def plots(self):
        """Subplots supported by `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`."""
        return self._plots

    @property
    def plots_defaults(self):
        """Defaults for `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots`."""
        return self._plots_defaults

    @property
    def Indicator(self):
        """Built indicator class."""
        return self._Indicator

    def from_custom_func(self,
                         custom_func: tp.Callable,
                         require_input_shape: bool = False,
                         param_settings: tp.KwargsLike = None,
                         in_output_settings: tp.KwargsLike = None,
                         hide_params: tp.Optional[tp.Sequence[str]] = None,
                         hide_default: bool = True,
                         var_args: bool = False,
                         keyword_only_args: bool = False,
                         **pipeline_kwargs) -> tp.Type[IndicatorBase]:
        """Build indicator class around a custom calculation function.

        In contrast to `IndicatorFactory.from_apply_func`, this method offers full flexbility.
        It's up to we to handle caching and concatenate columns for each parameter (for example,
        by using `vectorbtpro.base.combining.apply_and_concat`). Also, you must ensure that
        each output array has an appropriate number of columns, which is the number of columns in
        input arrays multiplied by the number of parameter combinations.

        Args:
            custom_func (callable): A function that takes broadcast arrays corresponding
                to `input_names`, broadcast in-place output arrays corresponding to `in_output_names`,
                broadcast parameter arrays corresponding to `param_names`, and other arguments and
                keyword arguments, and returns outputs corresponding to `output_names` and other objects
                that are then returned with the indicator instance.

                Can be Numba-compiled.

                !!! note
                    Shape of each output must be the same and match the shape of each input stacked
                    n times (= the number of parameter values) along the column axis.
            require_input_shape (bool): Whether to input shape is required.
            param_settings (dict): A dictionary of parameter settings keyed by name.
                See `run_pipeline` for keys.

                Can be overwritten by any run method.
            in_output_settings (dict): A dictionary of in-place output settings keyed by name.
                See `run_pipeline` for keys.

                Can be overwritten by any run method.
            hide_params (list of str): Parameter names to hide column levels for.

                Can be overwritten by any run method.
            hide_default (bool): Whether to hide column levels of parameters with default value.

                Can be overwritten by any run method.
            var_args (bool): Whether run methods should accept variable arguments (`*args`).

                Set to True if `custom_func` accepts positional agruments that are not listed in the config.
            keyword_only_args (bool): Whether run methods should accept keyword-only arguments (`*`).

                Set to True to force the user to use keyword arguments (e.g., to avoid misplacing arguments).
            **pipeline_kwargs: Keyword arguments passed to `run_pipeline`.

                Can be overwritten by any run method.

                Can contain default values and also references to other arguments wrapped
                with `vectorbtpro.base.reshaping.Ref`.

        Returns:
            `Indicator`, and optionally other objects that are returned by `custom_func`
            and exceed `output_names`.

        Usage:
            The following example produces the same indicator as the `IndicatorFactory.from_apply_func` example.

            ```pycon
            >>> @njit
            >>> def apply_func_nb(i, ts1, ts2, p1, p2, arg1, arg2):
            ...     return ts1 * p1[i] + arg1, ts2 * p2[i] + arg2

            >>> @njit
            ... def custom_func(ts1, ts2, p1, p2, arg1, arg2):
            ...     return vbt.base.combining.apply_and_concat_multiple_nb(
            ...         len(p1), apply_func_nb, ts1, ts2, p1, p2, arg1, arg2)

            >>> MyInd = vbt.IF(
            ...     input_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['o1', 'o2']
            ... ).from_custom_func(custom_func, var_args=True, arg2=200)

            >>> myInd = MyInd.run(price, price * 2, [1, 2], [3, 4], 100)
            >>> myInd.o1
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> myInd.o2
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  206.0  230.0  208.0  240.0
            2020-01-02  212.0  224.0  216.0  232.0
            2020-01-03  218.0  218.0  224.0  224.0
            2020-01-04  224.0  212.0  232.0  216.0
            2020-01-05  230.0  206.0  240.0  208.0
            ```

            The difference between `apply_func_nb` here and in `IndicatorFactory.from_apply_func` is that
            here it takes the index of the current parameter combination that can be used for parameter selection.
            You can also remove the entire `apply_func_nb` and define your logic in `custom_func`
            (which shouldn't necessarily be Numba-compiled):

            ```pycon
            >>> @njit
            ... def custom_func(ts1, ts2, p1, p2, arg1, arg2):
            ...     input_shape = ts1.shape
            ...     n_params = len(p1)
            ...     out1 = np.empty((input_shape[0], input_shape[1] * n_params), dtype=np.float_)
            ...     out2 = np.empty((input_shape[0], input_shape[1] * n_params), dtype=np.float_)
            ...     for k in range(n_params):
            ...         for col in range(input_shape[1]):
            ...             for i in range(input_shape[0]):
            ...                 out1[i, input_shape[1] * k + col] = ts1[i, col] * p1[k] + arg1
            ...                 out2[i, input_shape[1] * k + col] = ts2[i, col] * p2[k] + arg2
            ...     return out1, out2
            ```
        """
        Indicator = self.Indicator

        short_name = self.short_name
        prepend_name = self.prepend_name
        input_names = self.input_names
        param_names = self.param_names
        in_output_names = self.in_output_names
        output_names = self.output_names

        all_input_names = input_names + param_names + in_output_names

        setattr(Indicator, 'custom_func', custom_func)

        def _split_args(args: tp.Sequence) -> tp.Tuple[
            tp.Dict[str, tp.ArrayLike],
            tp.Dict[str, tp.ArrayLike],
            tp.Dict[str, tp.Params],
            tp.Args
        ]:
            inputs = dict(zip(input_names, args[:len(input_names)]))
            checks.assert_len_equal(inputs, input_names)
            args = args[len(input_names):]

            params = dict(zip(param_names, args[:len(param_names)]))
            checks.assert_len_equal(params, param_names)
            args = args[len(param_names):]

            in_outputs = dict(zip(in_output_names, args[:len(in_output_names)]))
            checks.assert_len_equal(in_outputs, in_output_names)
            args = args[len(in_output_names):]
            if not var_args and len(args) > 0:
                raise TypeError("Variable length arguments are not supported by this function "
                                "(var_args is set to False)")

            return inputs, in_outputs, params, args

        for k, v in pipeline_kwargs.items():
            if k in param_names and not isinstance(v, Default):
                pipeline_kwargs[k] = Default(v)  # track default params
        pipeline_kwargs = merge_dicts({k: None for k in in_output_names}, pipeline_kwargs)

        # Display default parameters and in-place outputs in the signature
        default_kwargs = {}
        for k in list(pipeline_kwargs.keys()):
            if k in input_names or k in param_names or k in in_output_names:
                default_kwargs[k] = pipeline_kwargs.pop(k)

        if var_args and keyword_only_args:
            raise ValueError("var_args and keyword_only_args cannot be used together")

        # Add private run method
        def_run_kwargs = dict(
            short_name=short_name,
            hide_params=hide_params,
            hide_default=hide_default,
            **default_kwargs
        )

        def _run(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunOutputT:
            _short_name = kwargs.pop('short_name', def_run_kwargs['short_name'])
            _hide_params = kwargs.pop('hide_params', def_run_kwargs['hide_params'])
            _hide_default = kwargs.pop('hide_default', def_run_kwargs['hide_default'])
            _param_settings = merge_dicts(
                param_settings,
                kwargs.pop('param_settings', {})
            )
            _in_output_settings = merge_dicts(
                in_output_settings,
                kwargs.pop('in_output_settings', {})
            )

            if _hide_params is None:
                _hide_params = []

            args = list(args)

            # Split arguments
            inputs, in_outputs, params, args = _split_args(args)

            # Prepare column levels
            level_names = []
            hide_levels = []
            for pname in param_names:
                level_name = _short_name + '_' + pname if prepend_name else pname
                level_names.append(level_name)
                if pname in _hide_params or (_hide_default and isinstance(params[pname], Default)):
                    hide_levels.append(level_name)
            for k, v in params.items():
                if isinstance(v, Default):
                    params[k] = v.value

            # Run the pipeline
            results = run_pipeline(
                len(output_names),  # number of returned outputs
                custom_func,
                *args,
                require_input_shape=require_input_shape,
                inputs=inputs,
                in_outputs=in_outputs,
                params=params,
                level_names=level_names,
                hide_levels=hide_levels,
                param_settings=_param_settings,
                in_output_settings=_in_output_settings,
                **merge_dicts(pipeline_kwargs, kwargs)
            )

            # Return the raw result if any of the flags are set
            if kwargs.get('return_raw', False) or kwargs.get('return_cache', False):
                return results

            # Unpack the result
            wrapper, \
            new_input_list, \
            input_mapper, \
            in_output_list, \
            output_list, \
            new_param_list, \
            mapper_list, \
            other_list = results

            # Create a new instance
            obj = cls(
                wrapper,
                new_input_list,
                input_mapper,
                in_output_list,
                output_list,
                new_param_list,
                mapper_list,
                short_name,
                tuple(level_names)
            )
            if len(other_list) > 0:
                return (obj, *tuple(other_list))
            return obj

        setattr(Indicator, '_run', classmethod(_run))

        # Add public run method
        # Create function dynamically to provide user with a proper signature
        def compile_run_function(func_name: str, docstring: str, _default_kwargs: tp.KwargsLike = None) -> tp.Callable:
            pos_names = []
            main_kw_names = []
            other_kw_names = []
            if _default_kwargs is None:
                _default_kwargs = {}
            for k in input_names + param_names:
                if k in _default_kwargs:
                    main_kw_names.append(k)
                else:
                    pos_names.append(k)
            main_kw_names.extend(in_output_names)  # in_output_names are keyword-only
            for k, v in _default_kwargs.items():
                if k not in pos_names and k not in main_kw_names:
                    other_kw_names.append(k)

            _0 = func_name
            _1 = '*, ' if keyword_only_args else ''
            _2 = []
            if require_input_shape:
                _2.append('input_shape')
            _2.extend(pos_names)
            _2 = ', '.join(_2) + ', ' if len(_2) > 0 else ''
            _3 = '*args, ' if var_args else ''
            _4 = ['{}={}'.format(k, k) for k in main_kw_names + other_kw_names]
            _4 = ', '.join(_4) + ', ' if len(_4) > 0 else ''
            _5 = docstring
            _6 = all_input_names
            _6 = ', '.join(_6) + ', ' if len(_6) > 0 else ''
            _7 = []
            if require_input_shape:
                _7.append('input_shape')
            _7.extend(other_kw_names)
            _7 = ['{}={}'.format(k, k) for k in _7]
            _7 = ', '.join(_7) + ', ' if len(_7) > 0 else ''
            func_str = "@classmethod\n" \
                       "def {0}(cls, {1}{2}{3}{4}**kwargs):\n" \
                       "    \"\"\"{5}\"\"\"\n" \
                       "    return cls._{0}({6}{3}{7}**kwargs)".format(
                _0, _1, _2, _3, _4, _5, _6, _7
            )
            scope = {**dict(Default=Default), **_default_kwargs}
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, 'single')
            exec(code, scope)
            return scope[func_name]

        _0 = self.class_name
        _1 = ''
        if len(self.input_names) > 0:
            _1 += '\n* Inputs: ' + ', '.join(map(lambda x: f'`{x}`', self.input_names))
        if len(self.in_output_names) > 0:
            _1 += '\n* In-place outputs: ' + ', '.join(map(lambda x: f'`{x}`', self.in_output_names))
        if len(self.param_names) > 0:
            _1 += '\n* Parameters: ' + ', '.join(map(lambda x: f'`{x}`', self.param_names))
        if len(self.output_names) > 0:
            _1 += '\n* Outputs: ' + ', '.join(map(lambda x: f'`{x}`', self.output_names))
        run_docstring = """Run `{0}` indicator.
{1}

Pass a list of parameter names as `hide_params` to hide their column levels.
Set `hide_default` to False to show the column levels of the parameters with a default value.

Other keyword arguments are passed to `vectorbtpro.indicators.factory.run_pipeline`.""".format(_0, _1)
        run = compile_run_function('run', run_docstring, def_run_kwargs)
        run.__qualname__ = f'{Indicator.__name__}.run'
        setattr(Indicator, 'run', run)

        if len(param_names) > 0:
            # Add private run_combs method
            def_run_combs_kwargs = dict(
                r=2,
                param_product=False,
                comb_func=itertools.combinations,
                run_unique=True,
                short_names=None,
                hide_params=hide_params,
                hide_default=hide_default,
                **default_kwargs
            )

            def _run_combs(cls: tp.Type[IndicatorBaseT], *args, **kwargs) -> RunCombsOutputT:
                _r = kwargs.pop('r', def_run_combs_kwargs['r'])
                _param_product = kwargs.pop('param_product', def_run_combs_kwargs['param_product'])
                _comb_func = kwargs.pop('comb_func', def_run_combs_kwargs['comb_func'])
                _run_unique = kwargs.pop('run_unique', def_run_combs_kwargs['run_unique'])
                _short_names = kwargs.pop('short_names', def_run_combs_kwargs['short_names'])
                _hide_params = kwargs.pop('hide_params', def_run_kwargs['hide_params'])
                _hide_default = kwargs.pop('hide_default', def_run_kwargs['hide_default'])
                _param_settings = merge_dicts(
                    param_settings,
                    kwargs.get('param_settings', {})
                )

                if _hide_params is None:
                    _hide_params = []
                if _short_names is None:
                    _short_names = [f'{short_name}_{str(i + 1)}' for i in range(_r)]

                args = list(args)

                # Split arguments
                inputs, in_outputs, params, args = _split_args(args)

                # Hide params
                for pname in param_names:
                    if _hide_default and isinstance(params[pname], Default):
                        params[pname] = params[pname].value
                        if pname not in _hide_params:
                            _hide_params.append(pname)
                checks.assert_len_equal(params, param_names)

                # Bring argument to list format
                input_list = list(inputs.values())
                in_output_list = list(in_outputs.values())
                param_list = list(params.values())

                # Prepare params
                for i, pname in enumerate(param_names):
                    is_tuple = _param_settings.get(pname, {}).get('is_tuple', False)
                    is_array_like = _param_settings.get(pname, {}).get('is_array_like', False)
                    param_list[i] = params_to_list(params[pname], is_tuple, is_array_like)
                if _param_product:
                    param_list = create_param_product(param_list)
                else:
                    param_list = broadcast_params(param_list)

                # Speed up by pre-calculating raw outputs
                if _run_unique:
                    raw_results = cls._run(
                        *input_list,
                        *param_list,
                        *in_output_list,
                        *args,
                        return_raw=True,
                        run_unique=False,
                        **kwargs
                    )
                    kwargs['use_raw'] = raw_results  # use them next time

                # Generate indicator instances
                instances = []
                if _comb_func == itertools.product:
                    param_lists = zip(*_comb_func(zip(*param_list), repeat=_r))
                else:
                    param_lists = zip(*_comb_func(zip(*param_list), _r))
                for i, param_list in enumerate(param_lists):
                    instances.append(cls._run(
                        *input_list,
                        *zip(*param_list),
                        *in_output_list,
                        *args,
                        short_name=_short_names[i],
                        hide_params=_hide_params,
                        hide_default=_hide_default,
                        run_unique=False,
                        **kwargs
                    ))
                return tuple(instances)

            setattr(Indicator, '_run_combs', classmethod(_run_combs))

            # Add public run_combs method
            _0 = self.class_name
            _1 = ''
            if len(self.input_names) > 0:
                _1 += '\n* Inputs: ' + ', '.join(map(lambda x: f'`{x}`', self.input_names))
            if len(self.in_output_names) > 0:
                _1 += '\n* In-place outputs: ' + ', '.join(map(lambda x: f'`{x}`', self.in_output_names))
            if len(self.param_names) > 0:
                _1 += '\n* Parameters: ' + ', '.join(map(lambda x: f'`{x}`', self.param_names))
            if len(self.output_names) > 0:
                _1 += '\n* Outputs: ' + ', '.join(map(lambda x: f'`{x}`', self.output_names))
            run_combs_docstring = """Create a combination of multiple `{0}` indicators using function `comb_func`.
{1}

`comb_func` must accept an iterable of parameter tuples and `r`. 
Also accepts all combinatoric iterators from itertools such as `itertools.combinations`.
Pass `r` to specify how many indicators to run. 
Pass `short_names` to specify the short name for each indicator. 
Set `run_unique` to True to first compute raw outputs for all parameters, 
and then use them to build each indicator (faster).

Other keyword arguments are passed to `{0}.run`.

!!! note
    This method should only be used when multiple indicators are needed. 
    To test multiple parameters, pass them as lists to `{0}.run`.
""".format(_0, _1)
            run_combs = compile_run_function('run_combs', run_combs_docstring, def_run_combs_kwargs)
            run_combs.__qualname__ = f'{Indicator.__name__}.run_combs'
            setattr(Indicator, 'run_combs', run_combs)

        return Indicator

    def from_apply_func(self,
                        apply_func: tp.Callable,
                        cache_func: tp.Optional[tp.Callable] = None,
                        pass_packed: bool = False,
                        kwargs_to_args: tp.Optional[tp.Sequence[str]] = None,
                        jitted_loop: bool = False,
                        remove_kwargs: tp.Optional[bool] = None,
                        **kwargs) -> tp.Type[IndicatorBase]:
        """Build indicator class around a custom apply function.

        In contrast to `IndicatorFactory.from_custom_func`, this method handles a lot of things for you,
        such as caching, parameter selection, and concatenation. Your part is writing a function `apply_func`
        that accepts a selection of parameters (single values as opposed to multiple values in
        `IndicatorFactory.from_custom_func`) and does the calculation. It then automatically concatenates
        the resulting arrays into a single array per output.

        While this approach is simpler, it's also less flexible, since we can only work with
        one parameter selection at a time and can't view all parameters.

        The execution and concatenation is performed using `vectorbtpro.base.combining.apply_and_concat`.

        !!! note
            If `apply_func` is a Numba-compiled function:

            * All inputs are automatically converted to NumPy arrays
            * Each argument in `*args` must be of a Numba-compatible type
            * You cannot pass keyword arguments
            * Your outputs must be arrays of the same shape, data type and data order

        Args:
            apply_func (callable): A function that takes inputs, selection of parameters, and
                other arguments, and does calculations to produce outputs.

                Arguments are passed to `apply_func` in the following order:

                * `input_shape` if `pass_input_shape` is set to True and `input_shape` not in `kwargs_to_args`
                * `col` if `per_column` and `pass_col` are set to True and `col` not in `kwargs_to_args`
                * broadcast time-series arrays corresponding to `input_names` (one-dimensional)
                * broadcast in-place output arrays corresponding to `in_output_names` (one-dimensional)
                * single parameter selection corresponding to `param_names` (single values)
                * variable arguments if `var_args` is set to True
                * arguments listed in `kwargs_to_args`
                * `flex_2d` if `pass_flex_2d` is set to True and `flex_2d` not in `kwargs_to_args`
                * keyword arguments if `apply_func` is not Numba-compiled

                Can be Numba-compiled.

                !!! note
                    Shape of each output must be the same and match the shape of each input.
            cache_func (callable): A caching function to preprocess data beforehand.

                Takes the same arguments as `apply_func`. Must return a single object or a tuple of objects.
                All returned objects will be passed unpacked as last arguments to `apply_func`.

                Can be Numba-compiled.
            pass_packed (bool): Whether to pass packed tuples for inputs, in-place outputs, and parameters.
            kwargs_to_args (list of str): Keyword arguments from `kwargs` dict to pass as
                positional arguments to the apply function.

                Should be used together with `jitted_loop` set to True since Numba doesn't support
                variable keyword arguments.

                Defaults to []. Order matters.
            jitted_loop (bool): Whether to loop using a jitter.

                Parameter selector will be automatically compiled using Numba.

                Set to True when iterating large number of times over small input,
                but note that Numba doesn't support variable keyword arguments.
            remove_kwargs (bool): Whether to remove keyword arguments when selecting parameters.

                If None, gets set to True if `jitted_loop` is True.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_custom_func`, all the way down
                to `vectorbtpro.base.combining.apply_and_concat`.

        Returns:
            Indicator

        Usage:
            The following example produces the same indicator as the `IndicatorFactory.from_custom_func` example.

            ```pycon
            >>> @njit
            ... def apply_func_nb(ts1, ts2, p1, p2, arg1, arg2):
            ...     return ts1 * p1 + arg1, ts2 * p2 + arg2

            >>> MyInd = vbt.IF(
            ...     input_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['out1', 'out2']
            ... ).from_apply_func(
            ...     apply_func_nb, var_args=True,
            ...     kwargs_to_args=['arg2'], arg2=200)

            >>> myInd = MyInd.run(price, price * 2, [1, 2], [3, 4], 100)
            >>> myInd.out1
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> myInd.out2
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  206.0  230.0  208.0  240.0
            2020-01-02  212.0  224.0  216.0  232.0
            2020-01-03  218.0  218.0  224.0  224.0
            2020-01-04  224.0  212.0  232.0  216.0
            2020-01-05  230.0  206.0  240.0  208.0
            ```

            To change the execution engine or specify other engine-related arguments, use `execute_kwargs`:

            ```pycon
            >>> import time

            >>> def apply_func(ts, p):
            ...     time.sleep(1)
            ...     return ts * p

            >>> MyInd = vbt.IF(
            ...     input_names=['ts'],
            ...     param_names=['p'],
            ...     output_names=['out']
            ... ).from_apply_func(apply_func)

            >>> %timeit MyInd.run(price, [1, 2, 3])
            3.02 s ± 3.47 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

            >>> %timeit MyInd.run(price, [1, 2, 3], execute_kwargs=dict(engine='dask'))
            1.02 s ± 2.67 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
            ```
        """
        Indicator = self.Indicator

        setattr(Indicator, 'apply_func', apply_func)

        module_name = self.module_name
        input_names = self.input_names
        output_names = self.output_names
        in_output_names = self.in_output_names
        param_names = self.param_names

        num_ret_outputs = len(output_names)

        if kwargs_to_args is None:
            kwargs_to_args = []
        if remove_kwargs is None:
            remove_kwargs = jitted_loop

        # Build a function that selects a parameter tuple
        # Do it here to avoid compilation with Numba every time custom_func is run
        _0 = "i"
        _0 += ", args_before"
        if len(input_names) > 0:
            _0 += ', ' + ', '.join(input_names)
        if len(in_output_names) > 0:
            _0 += ', ' + ', '.join(in_output_names)
        if len(param_names) > 0:
            _0 += ', ' + ', '.join(param_names)
        _0 += ", *args"
        if not remove_kwargs:
            _0 += ", **kwargs"
        _1 = "*args_before"
        if pass_packed:
            if len(input_names) > 0:
                _1 += ', (' + ', '.join(input_names) + ',)'
            else:
                _1 += ', ()'
            if len(in_output_names) > 0:
                _1 += ', (' + ', '.join(map(lambda x: x + '[i]', in_output_names)) + ',)'
            else:
                _1 += ', ()'
            if len(param_names) > 0:
                _1 += ', (' + ', '.join(map(lambda x: x + '[i]', param_names)) + ',)'
            else:
                _1 += ', ()'
        else:
            if len(input_names) > 0:
                _1 += ', ' + ', '.join(input_names)
            if len(in_output_names) > 0:
                _1 += ', ' + ', '.join(map(lambda x: x + '[i]', in_output_names))
            if len(param_names) > 0:
                _1 += ', ' + ', '.join(map(lambda x: x + '[i]', param_names))
        _1 += ", *args"
        if not remove_kwargs:
            _1 += ", **kwargs"
        func_str = "def select_params_func({0}):\n   return apply_func({1})".format(_0, _1)
        scope = {'apply_func': apply_func}
        filename = inspect.getfile(lambda: None)
        code = compile(func_str, filename, 'single')
        exec(code, scope)
        select_params_func = scope['select_params_func']
        if module_name is not None:
            select_params_func.__module__ = module_name
        if jitted_loop:
            select_params_func = njit(select_params_func)

        def custom_func(input_list: tp.List[tp.AnyArray],
                        in_output_list: tp.List[tp.List[tp.AnyArray]],
                        param_list: tp.List[tp.List[tp.Param]],
                        *args,
                        input_shape: tp.Optional[tp.Shape] = None,
                        col: tp.Optional[int] = None,
                        flex_2d: tp.Optional[bool] = None,
                        return_cache: bool = False,
                        use_cache: tp.Optional[CacheOutputT] = None,
                        **_kwargs) -> tp.Union[None, CacheOutputT, tp.Array2d, tp.List[tp.Array2d]]:
            """Custom function that forwards inputs and parameters to `apply_func`."""

            n_params = len(param_list[0]) if len(param_list) > 0 else 1
            args_before = ()
            if input_shape is not None and 'input_shape' not in kwargs_to_args:
                args_before += (input_shape,)
            if col is not None and 'col' not in kwargs_to_args:
                args_before += (col,)

            # Pass some keyword arguments as positional (required by numba)
            more_args = ()
            for key in kwargs_to_args:
                value = _kwargs.pop(key)  # important: remove from kwargs
                more_args += (value,)
            if flex_2d is not None and 'flex_2d' not in kwargs_to_args:
                more_args += (flex_2d,)

            # Caching
            cache = use_cache
            if cache is None and cache_func is not None:
                if checks.is_numba_func(cache_func):
                    _in_output_list = list(map(to_typed_list, in_output_list))
                    _param_list = list(map(to_typed_list, param_list))
                else:
                    _in_output_list = in_output_list
                    _param_list = param_list
                if pass_packed:
                    cache = cache_func(
                        *args_before,
                        input_list,
                        _in_output_list,
                        _param_list,
                        *args,
                        *more_args,
                        **_kwargs
                    )
                else:
                    cache = cache_func(
                        *args_before,
                        *input_list,
                        *_in_output_list,
                        *_param_list,
                        *args,
                        *more_args,
                        **_kwargs
                    )
            if return_cache:
                return cache
            if cache is None:
                cache = ()
            if not isinstance(cache, tuple):
                cache = (cache,)

            if jitted_loop:
                _in_output_list = list(map(to_typed_list, in_output_list))
                _param_list = list(map(to_typed_list, param_list))
            else:
                _in_output_list = in_output_list
                _param_list = param_list
            return combining.apply_and_concat(
                n_params,
                select_params_func,
                args_before,
                *input_list,
                *_in_output_list,
                *_param_list,
                *args,
                *more_args,
                *cache,
                **_kwargs,
                n_outputs=num_ret_outputs,
                jitted_loop=jitted_loop
            )

        return self.from_custom_func(custom_func, pass_packed=True, **kwargs)

    # ############# Expressions ############# #

    @class_or_instancemethod
    def from_expr(cls_or_self,
                  expr: str,
                  factory_kwargs: tp.KwargsLike = None,
                  parse_special_vars: bool = True,
                  magnet_input_names: tp.Iterable[str] = None,
                  func_mapping: tp.KwargsLike = None,
                  res_func_mapping: tp.KwargsLike = None,
                  use_pd_eval: bool = False,
                  pd_eval_kwargs: tp.KwargsLike = None,
                  **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class from an indicator expression.

        Args:
            expr (str): Expression.

                Expression must be a string with a valid Python code.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.

                Only applied when calling the class method.
            parse_special_vars (bool): Whether to parse variables starting with `@`.
            magnet_input_names (iterable of str): Names recognized as input names.

                Defaults to `open`, `high`, `low`, `close`, and `volume`.
            func_mapping (mapping): Mapping merged over `vectorbtpro.indicators.expr.expr_func_config`.

                Each key must be a function name and each value must be a dict with
                `func` and optionally `magnet_input_names`.
            res_func_mapping (mapping): Mapping merged over `vectorbtpro.indicators.expr.expr_res_func_config`.

                Each key must be a function name and each value must be a dict with
                `func` and optionally `magnet_input_names`.
            use_pd_eval (bool): Whether to use `pd.eval`.

                Otherwise, uses `vectorbtpro.utils.eval_.multiline_eval`.

                !!! hint
                    By default, operates on NumPy objects using NumExpr.
                    If you want to operate on Pandas objects, set `keep_pd` to True.
            pd_eval_kwargs (dict): Keyword arguments passed to `pd.eval`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_apply_func`.

        Returns:
            Indicator

        Searches each variable name parsed from `expr` in

        * `vectorbtpro.indicators.expr.expr_res_func_config` (calls right away),
        * `vectorbtpro.indicators.expr.expr_func_config`,
        * inputs, in-outputs, and params,
        * keyword arguments,
        * attributes of `np`,
        * attributes of `vectorbtpro.generic.nb` (with and without `_nb` suffix),
        * attributes of `vbt`, and
        * packages and modules.

        `vectorbtpro.indicators.expr.expr_func_config` and `vectorbtpro.indicators.expr.expr_res_func_config`
        can be overridden with `func_mapping` and `res_func_mapping` respectively.

        !!! note
            Each variable name is case-sensitive.

        When using the class method, all names are parsed from the expression itself.
        If any of `open`, `high`, `low`, `close`, and `volume` appear in the expression or
        in `magnet_input_names` in either `vectorbtpro.indicators.expr.expr_func_config` or
        `vectorbtpro.indicators.expr.expr_res_func_config`, they are automatically added to `input_names`.
        Set `magnet_input_names` to an empty list to disable this logic.

        If `parse_special_vars` is True, variables that start with `@` have a special meaning:

        * `@in_*`: input
        * `@inout_*`: in-output
        * `@p_*`: parameter

        !!! note
            The parsed names come in the same order they appear in the expression, not in the parsing order.

        The number of outputs is derived based on the number of commas outside of any bracket pair.
        If there is only one output, the output name is `out`. If more - `out1`, `out2`, etc.

        Any information can be overridden using `factory_kwargs`.

        Usage:
            ```pycon
            >>> WMA = vbt.IF(
            ...     class_name='WMA',
            ...     input_names=['close'],
            ...     param_names=['window'],
            ...     output_names=['wma']
            ... ).from_expr("wm_mean_nb(close, window)")

            >>> wma = WMA.run(price, window=[2, 3])
            >>> wma.wma
            wma_window                   2                   3
                               a         b         a         b
            2020-01-01       NaN       NaN       NaN       NaN
            2020-01-02  1.666667  4.333333       NaN       NaN
            2020-01-03  2.666667  3.333333  2.333333  3.666667
            2020-01-04  3.666667  2.333333  3.333333  2.666667
            2020-01-05  4.666667  1.333333  4.333333  1.666667
            ```

            The same can be achieved by calling the class method and providing prefixes
            to the variable names to indicate their type:

            ```pycon
            >>> expr = "wm_mean_nb((@in_high + @in_low) / 2, @p_window)"
            >>> WMA = vbt.IF.from_expr(expr)
            >>> wma = WMA.run(price + 1, price, window=[2, 3])
            >>> wma.out
            custom_window                   2                   3
                                  a         b         a         b
            2020-01-01          NaN       NaN       NaN       NaN
            2020-01-02     2.166667  4.833333       NaN       NaN
            2020-01-03     3.166667  3.833333  2.833333  4.166667
            2020-01-04     4.166667  2.833333  3.833333  3.166667
            2020-01-05     5.166667  1.833333  4.833333  2.166667
            ```

            Common (lower-case) input names from OHLCV are recognized automatically:

            ```pycon
            >>> expr = "wm_mean_nb((high + low) / 2, @p_window)"
            >>> WMA = vbt.IF.from_expr(expr)
            >>> wma = WMA.run(price + 1, price, window=[2, 3])
            >>> wma.out
            custom_window                   2                   3
                                  a         b         a         b
            2020-01-01          NaN       NaN       NaN       NaN
            2020-01-02     2.166667  4.833333       NaN       NaN
            2020-01-03     3.166667  3.833333  2.833333  4.166667
            2020-01-04     4.166667  2.833333  3.833333  3.166667
            2020-01-05     5.166667  1.833333  4.833333  2.166667
            ```
        """
        expr = expr.strip()
        if expr.endswith(','):
            expr = expr[:-1]
        if expr.startswith('(') and expr.endswith(')'):
            n_open_brackets = 0
            remove_brackets = True
            for i, s in enumerate(expr):
                if s == '(':
                    n_open_brackets += 1
                elif s == ')':
                    n_open_brackets -= 1
                    if n_open_brackets == 0 and i < len(expr) - 1:
                        remove_brackets = False
                        break
            if remove_brackets:
                expr = expr[1:-1]
        if expr.endswith(','):
            expr = expr[:-1]  # again

        func_mapping = merge_dicts(expr_func_config, func_mapping)
        res_func_mapping = merge_dicts(expr_res_func_config, res_func_mapping)

        if isinstance(cls_or_self, type):
            if magnet_input_names is None:
                magnet_input_names = ['open', 'high', 'low', 'close', 'volume']
            found_magnet_input_names = []
            input_names = []
            in_output_names = []
            param_names = []

            if parse_special_vars:
                for var_name in re.findall(r"@\w+", expr):
                    var_name = var_name.replace('@', '')
                    if var_name.startswith('in_'):
                        var_name = var_name[3:]
                        if var_name in magnet_input_names:
                            if var_name not in found_magnet_input_names:
                                found_magnet_input_names.append(var_name)
                        else:
                            if var_name not in input_names:
                                input_names.append(var_name)
                    elif var_name.startswith('inout_'):
                        var_name = var_name[6:]
                        if var_name not in in_output_names:
                            in_output_names.append(var_name)
                    elif var_name.startswith('p_'):
                        var_name = var_name[2:]
                        if var_name not in param_names:
                            param_names.append(var_name)

                expr = expr.replace("@in_", "__in_")
                expr = expr.replace("@inout_", "__inout_")
                expr = expr.replace("@p_", "__p_")

            var_names = get_expr_var_names(expr)
            for input_name in magnet_input_names:
                if input_name not in found_magnet_input_names:
                    if input_name in var_names:
                        found_magnet_input_names.append(input_name)
                        continue
                    for var_name in var_names:
                        if input_name in func_mapping.get(var_name, {}).get('magnet_input_names', []) or \
                                input_name in res_func_mapping.get(var_name, {}).get('magnet_input_names', []):
                            found_magnet_input_names.append(input_name)
                            break
            for input_name in magnet_input_names:
                if input_name in found_magnet_input_names:
                    input_names.append(input_name)

            n_open_brackets = 0
            n_outputs = 1
            for i, s in enumerate(expr):
                if s == ',' and n_open_brackets == 0:
                    n_outputs += 1
                elif s in '([{':
                    n_open_brackets += 1
                elif s in ')]}':
                    n_open_brackets -= 1
            if n_open_brackets != 0:
                raise ValueError("Couldn't parse the number of outputs: mismatching brackets")
            if n_outputs == 1:
                output_names = ['out']
            else:
                output_names = ['out%d' % (i + 1) for i in range(n_outputs)]

            factory = cls_or_self(
                **merge_dicts(
                    dict(
                        input_names=input_names,
                        in_output_names=in_output_names,
                        param_names=param_names,
                        output_names=output_names
                    ),
                    factory_kwargs
                )
            )
        else:
            factory = cls_or_self

        Indicator = factory.Indicator

        input_names = factory.input_names
        in_output_names = factory.in_output_names
        param_names = factory.param_names

        def apply_func(input_tuple: tp.Tuple[tp.AnyArray],
                       in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
                       param_tuple: tp.Tuple[tp.Param, ...],
                       **_kwargs) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            import vectorbtpro as vbt

            input_context = dict(
                np=np,
                pd=pd,
                vbt=vbt
            )
            for i, input in enumerate(input_tuple):
                input_context[input_names[i]] = input
            for i, in_output in enumerate(in_output_tuple):
                input_context[in_output_names[i]] = in_output
            for i, param in enumerate(param_tuple):
                input_context[param_names[i]] = param
            merged_context = merge_dicts(input_context, _kwargs)
            context = {}
            subbed_context = {}

            for var_name in get_expr_var_names(expr):
                if var_name in context:
                    continue
                if var_name.startswith('__in_'):
                    var = merged_context[var_name[5:]]
                elif var_name.startswith('__inout_'):
                    var = merged_context[var_name[8:]]
                elif var_name.startswith('__p_'):
                    var = merged_context[var_name[4:]]
                elif var_name in res_func_mapping:
                    var = res_func_mapping[var_name]['func']
                elif var_name in func_mapping:
                    var = func_mapping[var_name]['func']
                elif var_name in merged_context:
                    var = merged_context[var_name]
                elif hasattr(np, var_name):
                    var = getattr(np, var_name)
                elif hasattr(generic_nb, var_name):
                    var = getattr(generic_nb, var_name)
                elif hasattr(generic_nb, var_name + '_nb'):
                    var = getattr(generic_nb, var_name + '_nb')
                elif hasattr(vbt, var_name):
                    var = getattr(vbt, var_name)
                else:
                    try:
                        var = importlib.import_module(var_name)
                    except ModuleNotFoundError:
                        continue
                try:
                    if callable(var) and 'context' in get_func_arg_names(var):
                        var = functools.partial(var, context=merged_context)
                except:
                    pass
                if var_name in res_func_mapping:
                    var = var()
                context[var_name] = var

            if use_pd_eval:
                return pd.eval(expr, local_dict=context, **resolve_dict(pd_eval_kwargs))
            return multiline_eval(expr, context=context)

        return factory.from_apply_func(
            apply_func,
            pass_packed=True,
            pass_wrapper=True,
            **kwargs
        )

    @classmethod
    def from_wqa101(cls, alpha_idx: int, **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class from one of the WorldQuant's 101 alpha expressions.

        See `vectorbtpro.indicators.expr.wqa101_expr_config`.

        !!! note
            Some expressions that utilize cross-sectional operations require columns to be
            a multi-index with a level `sector`, `subindustry`, or `industry`.

        Usage:
            ```pycon
            >>> data = vbt.YFData.fetch(['BTC-USD', 'ETH-USD'])

            >>> WQA1 = vbt.IF.from_wqa101(1)
            >>> wqa1 = WQA1.run(data.get('Close'))
            >>> wqa1.out
            symbol                     BTC-USD  ETH-USD
            Date
            2014-09-17 00:00:00+00:00     0.25     0.25
            2014-09-18 00:00:00+00:00     0.25     0.25
            2014-09-19 00:00:00+00:00     0.25     0.25
            2014-09-20 00:00:00+00:00     0.25     0.25
            2014-09-21 00:00:00+00:00     0.25     0.25
            ...                            ...      ...
            2022-01-21 00:00:00+00:00     0.00     0.50
            2022-01-22 00:00:00+00:00     0.00     0.50
            2022-01-23 00:00:00+00:00     0.25     0.25
            2022-01-24 00:00:00+00:00     0.50     0.00
            2022-01-25 00:00:00+00:00     0.50     0.00

            [2688 rows x 2 columns]
            ```

            To get help on running the indicator, use the `help` command:

            ```pycon
            >>> help(WQA1.run)
            Help on method run:

            run(close, short_name='wqa1', hide_params=None, hide_default=True, **kwargs) method of vectorbtpro.generic.analyzable.MetaAnalyzable instance
                Run `WQA1` indicator.

                * Inputs: `close`
                * Outputs: `out`

                Pass a list of parameter names as `hide_params` to hide their column levels.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `vectorbtpro.indicators.factory.run_pipeline`.
            ```
        """
        return cls.from_expr(
            wqa101_expr_config[alpha_idx],
            factory_kwargs=dict(
                class_name="WQA%d" % alpha_idx,
                module_name=__name__ + '.wqa'
            ),
            **kwargs
        )

    # ############# Third party ############# #

    @classmethod
    def get_talib_indicators(cls) -> tp.Set[str]:
        """Get all TA-Lib indicators."""
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('talib')
        import talib

        return set(talib.get_functions())

    @classmethod
    def from_talib(cls, func_name: str, factory_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a TA-Lib function.

        Requires [TA-Lib](https://github.com/mrjbq7/ta-lib) installed.

        For input, parameter and output names, see [docs](https://github.com/mrjbq7/ta-lib/blob/master/docs/index.md).

        Args:
            func_name (str): Function name.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_custom_func`.

        Returns:
            Indicator

        Usage:
            ```pycon
            >>> SMA = vbt.IF.from_talib('SMA')

            >>> sma = SMA.run(price, timeperiod=[2, 3])
            >>> sma.real
            sma_timeperiod         2         3
                              a    b    a    b
            2020-01-01      NaN  NaN  NaN  NaN
            2020-01-02      1.5  4.5  NaN  NaN
            2020-01-03      2.5  3.5  2.0  4.0
            2020-01-04      3.5  2.5  3.0  3.0
            2020-01-05      4.5  1.5  4.0  2.0
            ```

            To get help on running the indicator, use the `help` command:

            ```pycon
            >>> help(SMA.run)
            Help on method run:

            run(close, timeperiod=30, short_name='sma', hide_params=None, hide_default=True, **kwargs) method of vectorbtpro.generic.analyzable.MetaAnalyzable instance
                Run `SMA` indicator.

                * Inputs: `close`
                * Parameters: `timeperiod`
                * Outputs: `real`

                Pass a list of parameter names as `hide_params` to hide their column levels.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `vectorbtpro.indicators.factory.run_pipeline`.
            ```
        """
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('talib')
        import talib
        from talib import abstract

        func_name = func_name.upper()
        talib_func = getattr(talib, func_name)
        info = abstract.Function(func_name)._Function__info
        input_names = []
        for in_names in info['input_names'].values():
            if isinstance(in_names, (list, tuple)):
                input_names.extend(list(in_names))
            else:
                input_names.append(in_names)
        class_name = info['name']
        class_docstring = "{}, {}".format(info['display_name'], info['group'])
        param_names = list(info['parameters'].keys())
        output_names = info['output_names']
        output_flags = info['output_flags']

        def apply_func(input_tuple: tp.Tuple[tp.AnyArray],
                       in_output_tuple: tp.Tuple[tp.AnyArray, ...],
                       param_tuple: tp.Tuple[tp.Param, ...],
                       **_kwargs) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            # TA-Lib functions can only process 1-dim arrays
            n_input_cols = input_tuple[0].shape[1]
            outputs = []
            for col in range(n_input_cols):
                output = talib_func(
                    *map(lambda x: x[:, col], input_tuple),
                    *param_tuple,
                    **_kwargs
                )
                outputs.append(output)
            if isinstance(outputs[0], tuple):  # multiple outputs
                outputs = list(zip(*outputs))
                return list(map(np.column_stack, outputs))
            return np.column_stack(outputs)

        TALibIndicator = cls(
            **merge_dicts(
                dict(
                    class_name=class_name,
                    class_docstring=class_docstring,
                    module_name=__name__ + '.talib',
                    input_names=input_names,
                    param_names=param_names,
                    output_names=output_names,
                    output_flags=output_flags
                ),
                factory_kwargs
            )
        ).from_apply_func(
            apply_func,
            pass_packed=True,
            **info['parameters'],
            **kwargs
        )
        return TALibIndicator

    @classmethod
    def parse_pandas_ta_config(cls,
                               func: tp.Callable,
                               test_input_names: tp.Optional[tp.Sequence[str]] = None,
                               test_index_len: int = 100,
                               **kwargs) -> tp.Kwargs:
        """Get the config of a pandas-ta indicator."""
        if test_input_names is None:
            test_input_names = {'open_', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'split'}

        input_names = []
        param_names = []
        defaults = {}
        output_names = []

        # Parse the function signature of the indicator to get input names
        sig = inspect.signature(func)
        for k, v in sig.parameters.items():
            if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                if v.annotation != inspect.Parameter.empty and v.annotation == pd.Series:
                    input_names.append(k)
                elif k in test_input_names:
                    input_names.append(k)
                elif v.default == inspect.Parameter.empty:
                    # Any positional argument is considered input
                    input_names.append(k)
                else:
                    param_names.append(k)
                    defaults[k] = v.default

        # To get output names, we need to run the indicator
        test_df = pd.DataFrame(
            {c: np.random.uniform(1, 10, size=(test_index_len,)) for c in input_names},
            index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(test_index_len)]
        )
        new_args = merge_dicts({c: test_df[c] for c in input_names}, kwargs)
        try:
            result = func(**new_args)
        except Exception as e:
            raise ValueError("Couldn't parse the indicator: " + str(e))

        # Concatenate Series/DataFrames if the result is a tuple
        if isinstance(result, tuple):
            results = []
            for i, r in enumerate(result):
                if not pd.Index.equals(r.index, test_df.index):
                    warnings.warn(f"Couldn't parse the output at index {i}: mismatching index", stacklevel=2)
                else:
                    results.append(r)
            if len(results) > 1:
                result = pd.concat(results, axis=1)
            elif len(results) == 1:
                result = results[0]
            else:
                raise ValueError("Couldn't parse the output")

        # Test if the produced array has the same index length
        if not pd.Index.equals(result.index, test_df.index):
            raise ValueError("Couldn't parse the output: mismatching index")

        # Standardize output names: remove numbers, remove hyphens, and bring to lower case
        output_cols = result.columns.tolist() if isinstance(result, pd.DataFrame) else [result.name]
        new_output_cols = []
        for i in range(len(output_cols)):
            name_parts = []
            for name_part in output_cols[i].split('_'):
                try:
                    float(name_part)
                    continue
                except:
                    name_parts.append(name_part.replace('-', '_').lower())
            output_col = '_'.join(name_parts)
            new_output_cols.append(output_col)

        # Add numbers to duplicates
        for k, v in Counter(new_output_cols).items():
            if v == 1:
                output_names.append(k)
            else:
                for i in range(v):
                    output_names.append(k + str(i))

        return dict(
            class_name=func.__name__.upper(),
            class_docstring=func.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults
        )

    @classmethod
    def get_pandas_ta_indicators(cls, silence_warnings: bool = True, **kwargs) -> tp.Set[str]:
        """Get all pandas-ta indicators.

        !!! note
            Returns only the indicators that have been successfully parsed."""
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('pandas_ta')
        import pandas_ta

        indicators = set()
        for func_name in [_k for k, v in pandas_ta.Category.items() for _k in v]:
            try:
                cls.parse_pandas_ta_config(getattr(pandas_ta, func_name), **kwargs)
                indicators.add(func_name.upper())
            except Exception as e:
                if not silence_warnings:
                    warnings.warn(f"Function {func_name}: " + str(e), stacklevel=2)
        return indicators

    @classmethod
    def from_pandas_ta(cls, func_name: str, parse_kwargs: tp.KwargsLike = None,
                       factory_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a pandas-ta function.

        Requires [pandas-ta](https://github.com/twopirllc/pandas-ta) installed.

        Args:
            func_name (str): Function name.
            parse_kwargs (dict): Keyword arguments passed to `IndicatorFactory.parse_pandas_ta_config`.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_custom_func`.

        Returns:
            Indicator

        Usage:
            ```pycon
            >>> SMA = vbt.IF.from_pandas_ta('SMA')

            >>> sma = SMA.run(price, length=[2, 3])
            >>> sma.sma
            sma_length         2         3
                          a    b    a    b
            2020-01-01  NaN  NaN  NaN  NaN
            2020-01-02  1.5  4.5  NaN  NaN
            2020-01-03  2.5  3.5  2.0  4.0
            2020-01-04  3.5  2.5  3.0  3.0
            2020-01-05  4.5  1.5  4.0  2.0
            ```

            To get help on running the indicator, use the `help` command:

            ```pycon
            >>> help(SMA.run)
            Help on method run:

            run(close, length=None, offset=None, short_name='sma', hide_params=None, hide_default=True, **kwargs) method of vectorbtpro.generic.analyzable.MetaAnalyzable instance
                Run `SMA` indicator.

                * Inputs: `close`
                * Parameters: `length`, `offset`
                * Outputs: `sma`

                Pass a list of parameter names as `hide_params` to hide their column levels.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `vectorbtpro.indicators.factory.run_pipeline`.
            ```

            To get the indicator docstring, use the `help` command or print the `__doc__` attribute:

            ```pycon
            >>> print(SMA.__doc__)
            Simple Moving Average (SMA)

            The Simple Moving Average is the classic moving average that is the equally
            weighted average over n periods.

            Sources:
                https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

            Calculation:
                Default Inputs:
                    length=10
                SMA = SUM(close, length) / length

            Args:
                close (pd.Series): Series of 'close's
                length (int): It's period. Default: 10
                offset (int): How many periods to offset the result. Default: 0

            Kwargs:
                adjust (bool): Default: True
                presma (bool, optional): If True, uses SMA for initial value.
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method

            Returns:
                pd.Series: New feature generated.
            ```
        """
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('pandas_ta')
        import pandas_ta

        func_name = func_name.lower()
        pandas_ta_func = getattr(pandas_ta, func_name)

        if parse_kwargs is None:
            parse_kwargs = {}
        config = cls.parse_pandas_ta_config(pandas_ta_func, **parse_kwargs)

        def apply_func(input_tuple: tp.Tuple[tp.AnyArray],
                       in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
                       param_tuple: tp.Tuple[tp.Param, ...],
                       **_kwargs) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            is_series = isinstance(input_tuple[0], pd.Series)
            n_input_cols = 1 if is_series else len(input_tuple[0].columns)
            outputs = []
            for col in range(n_input_cols):
                output = pandas_ta_func(
                    **{
                        name: input_tuple[i] if is_series else input_tuple[i].iloc[:, col]
                        for i, name in enumerate(config['input_names'])
                    },
                    **{
                        name: param_tuple[i]
                        for i, name in enumerate(config['param_names'])
                    },
                    **_kwargs
                )
                if isinstance(output, tuple):
                    _outputs = []
                    for o in output:
                        if pd.Index.equals(input_list[0].index, o.index):
                            _outputs.append(o)
                    if len(_outputs) > 1:
                        output = pd.concat(_outputs, axis=1)
                    elif len(_outputs) == 1:
                        output = _outputs[0]
                    else:
                        raise ValueError("No valid outputs were returned")
                if isinstance(output, pd.DataFrame):
                    output = tuple([output.iloc[:, i] for i in range(len(output.columns))])
                outputs.append(output)
            if isinstance(outputs[0], tuple):  # multiple outputs
                outputs = list(zip(*outputs))
                return list(map(np.column_stack, outputs))
            return np.column_stack(outputs)

        defaults = config.pop('defaults')
        PTAIndicator = cls(
            **merge_dicts(
                dict(module_name=__name__ + '.pandas_ta'),
                config,
                factory_kwargs
            )
        ).from_apply_func(
            apply_func,
            pass_packed=True,
            keep_pd=True,
            to_2d=False,
            **defaults,
            **kwargs
        )
        return PTAIndicator

    @classmethod
    def get_ta_indicators(cls) -> tp.Set[str]:
        """Get all ta indicators."""
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('ta')
        import ta

        ta_module_names = [k for k in dir(ta) if isinstance(getattr(ta, k), ModuleType)]
        indicators = set()
        for module_name in ta_module_names:
            module = getattr(ta, module_name)
            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) \
                        and obj != ta.utils.IndicatorMixin \
                        and issubclass(obj, ta.utils.IndicatorMixin):
                    indicators.add(obj.__name__)
        return indicators

    @classmethod
    def find_ta_indicator(cls, cls_name: str) -> IndicatorMixinT:
        """Get ta indicator class by its name."""
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('ta')
        import ta

        ta_module_names = [k for k in dir(ta) if isinstance(getattr(ta, k), ModuleType)]
        for module_name in ta_module_names:
            module = getattr(ta, module_name)
            if cls_name in dir(module):
                return getattr(module, cls_name)
        raise ValueError(f"Indicator \"{cls_name}\" not found")

    @classmethod
    def parse_ta_config(cls, ind_cls: IndicatorMixinT) -> tp.Kwargs:
        """Get the config of a ta indicator."""
        input_names = []
        param_names = []
        defaults = {}
        output_names = []

        # Parse the __init__ signature of the indicator class to get input names
        sig = inspect.signature(ind_cls)
        for k, v in sig.parameters.items():
            if v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD):
                if v.annotation == inspect.Parameter.empty:
                    raise ValueError(f"Argument \"{k}\" has no annotation")
                if v.annotation == pd.Series:
                    input_names.append(k)
                else:
                    param_names.append(k)
                    if v.default != inspect.Parameter.empty:
                        defaults[k] = v.default

        # Get output names by looking into instance methods
        for attr in dir(ind_cls):
            if not attr.startswith('_'):
                if inspect.signature(getattr(ind_cls, attr)).return_annotation == pd.Series:
                    output_names.append(attr)
                elif 'Returns:\n            pandas.Series' in getattr(ind_cls, attr).__doc__:
                    output_names.append(attr)

        return dict(
            class_name=ind_cls.__name__,
            class_docstring=ind_cls.__doc__,
            input_names=input_names,
            param_names=param_names,
            output_names=output_names,
            defaults=defaults
        )

    @classmethod
    def from_ta(cls, cls_name: str, factory_kwargs: tp.KwargsLike = None, **kwargs) -> tp.Type[IndicatorBase]:
        """Build an indicator class around a ta class.

        Requires [ta](https://github.com/bukosabino/ta) installed.

        Args:
            cls_name (str): Class name.
            factory_kwargs (dict): Keyword arguments passed to `IndicatorFactory`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_custom_func`.

        Returns:
            Indicator

        Usage:
            ```pycon
            >>> SMAIndicator = vbt.IF.from_ta('SMAIndicator')

            >>> sma = SMAIndicator.run(price, window=[2, 3])
            >>> sma.sma_indicator
            smaindicator_window    2         3
                                   a    b    a    b
            2020-01-01           NaN  NaN  NaN  NaN
            2020-01-02           1.5  4.5  NaN  NaN
            2020-01-03           2.5  3.5  2.0  4.0
            2020-01-04           3.5  2.5  3.0  3.0
            2020-01-05           4.5  1.5  4.0  2.0
            ```

            To get help on running the indicator, use the `help` command:

            ```pycon
            >>> help(SMAIndicator.run)
            Help on method run:

            run(close, window, fillna=False, short_name='smaindicator', hide_params=None, hide_default=True, **kwargs) method of vectorbtpro.generic.analyzable.MetaAnalyzable instance
                Run `SMAIndicator` indicator.

                * Inputs: `close`
                * Parameters: `window`, `fillna`
                * Outputs: `sma_indicator`

                Pass a list of parameter names as `hide_params` to hide their column levels.
                Set `hide_default` to False to show the column levels of the parameters with a default value.

                Other keyword arguments are passed to `vectorbtpro.indicators.factory.run_pipeline`.
            ```

            To get the indicator docstring, use the `help` command or print the `__doc__` attribute:

            ```pycon
            >>> print(SMAIndicator.__doc__)
            SMA - Simple Moving Average

                Args:
                    close(pandas.Series): dataset 'Close' column.
                    window(int): n period.
                    fillna(bool): if True, fill nan values.
            ```
        """
        from vectorbtpro.utils.opt_packages import assert_can_import
        assert_can_import('ta')

        ind_cls = cls.find_ta_indicator(cls_name)
        config = cls.parse_ta_config(ind_cls)

        def apply_func(input_tuple: tp.Tuple[tp.AnyArray],
                       in_output_tuple: tp.Tuple[tp.SeriesFrame, ...],
                       param_tuple: tp.Tuple[tp.Param, ...],
                       **_kwargs) -> tp.Union[tp.Array2d, tp.List[tp.Array2d]]:
            is_series = isinstance(input_tuple[0], pd.Series)
            n_input_cols = 1 if is_series else len(input_tuple[0].columns)
            outputs = []
            for col in range(n_input_cols):
                ind = ind_cls(
                    **{
                        name: input_tuple[i] if is_series else input_tuple[i].iloc[:, col]
                        for i, name in enumerate(config['input_names'])
                    },
                    **{
                        name: param_tuple[i]
                        for i, name in enumerate(config['param_names'])
                    },
                    **_kwargs
                )
                output = []
                for output_name in config['output_names']:
                    output.append(getattr(ind, output_name)())
                if len(output) == 1:
                    output = output[0]
                else:
                    output = tuple(output)
                outputs.append(output)
            if isinstance(outputs[0], tuple):  # multiple outputs
                outputs = list(zip(*outputs))
                return list(map(np.column_stack, outputs))
            return np.column_stack(outputs)

        defaults = config.pop('defaults')
        TAIndicator = cls(
            **merge_dicts(
                dict(module_name=__name__ + '.ta'),
                config,
                factory_kwargs
            )
        ).from_apply_func(
            apply_func,
            pass_packed=True,
            keep_pd=True,
            to_2d=False,
            **defaults,
            **kwargs
        )
        return TAIndicator
