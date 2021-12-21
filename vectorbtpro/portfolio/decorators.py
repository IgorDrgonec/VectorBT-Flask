# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Class decorators for portfolio."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import Config, merge_dicts, resolve_dict
from vectorbtpro.utils.decorators import cacheable_property, cached_property
from vectorbtpro.utils.parsing import get_func_arg_names


def attach_returns_acc_methods(config: Config) -> tp.ClassWrapper:
    """Class decorator to attach returns accessor methods.

    `config` must contain target method names (keys) and settings (values) with the following keys:

    * `source_name`: Name of the source method. Defaults to the target name.
    * `docstring`: Method docstring.

    The class must be a subclass of `vectorbtpro.portfolio.base.Portfolio`."""

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Portfolio")

        for target_name, settings in config.items():
            source_name = settings.get('source_name', target_name)
            docstring = settings.get('docstring', f"See `vectorbtpro.returns.accessors.ReturnsAccessor.{source_name}`.")

            def new_method(self,
                           *,
                           group_by: tp.GroupByLike = None,
                           benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                           freq: tp.Optional[tp.FrequencyLike] = None,
                           year_freq: tp.Optional[tp.FrequencyLike] = None,
                           use_asset_returns: bool = False,
                           jitted: tp.JittedOption = None,
                           _source_name: str = source_name,
                           **kwargs) -> tp.Any:
                returns_acc = self.get_returns_acc(
                    group_by=group_by,
                    benchmark_rets=benchmark_rets,
                    freq=freq,
                    year_freq=year_freq,
                    use_asset_returns=use_asset_returns,
                    jitted=jitted
                )
                ret_method = getattr(returns_acc, _source_name)
                if 'jitted' in get_func_arg_names(ret_method):
                    kwargs['jitted'] = jitted
                return ret_method(**kwargs)

            new_method.__name__ = 'get_' + target_name
            new_method.__qualname__ = f"{cls.__name__}.get_{target_name}"
            new_method.__doc__ = docstring
            setattr(cls, new_method.__name__, new_method)
        return cls

    return wrapper


def attach_shortcut_properties(config: Config) -> tp.ClassWrapper:
    """Class decorator to attach shortcut properties.

    `config` must contain target property names (keys) and settings (values) with the following keys:

    * `method_name`: Name of the source method. Defaults to the target name prepended with the prefix `get_`.
    * `use_in_outputs`: Whether the property can return an in-place output. Defaults to True.
    * `field_aliases`: Fields to search for in `vectorbtpro.portfolio.base.Portfolio.in_outputs`.
    * `obj_type`: Type of the returned object. Can be 'array' for 2-dim arrays, 'red_array' for 1-dim arrays,
        and 'records' for record arrays. Defaults to 'array'.
    * `group_aware`: Whether the returned object is aligned based on the current grouping.
        Defaults to True.
    * `wrap_kwargs`: Keyword arguments passed to `vectorbtpro.base.wrapping.ArrayWrapper.wrap` and
        `vectorbtpro.base.wrapping.ArrayWrapper.wrap_reduced`. Defaults to None.
    * `wrap_func`: Wrapping function. Defaults to None.
    * `method_kwargs`: Keyword arguments passed to the source method. Defaults to None.
    * `decorator`: Defaults to `vectorbtpro.utils.decorators.cached_property` for object types
        'records' and 'red_array'. Otherwise, to `vectorbtpro.utils.decorators.cacheable_property`.
    * `decorator_kwargs`: Keyword arguments passed to the decorator. By default,
        includes options `obj_type` and `group_aware`.
    * `docstring`: Method docstring.

    The class must be a subclass of `vectorbtpro.portfolio.base.Portfolio`."""

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        checks.assert_subclass_of(cls, "Portfolio")

        for target_name, settings in config.items():
            if target_name.startswith('get_'):
                raise ValueError(f"Property names cannot have prefix 'get_' ('{target_name}')")
            method_name = settings.get('method_name', 'get_' + target_name)
            use_in_outputs = settings.get('use_in_outputs', True)
            field_aliases = [target_name, *settings.get('field_aliases', [])]
            obj_type = settings.get('obj_type', 'array')
            group_by_aware = settings.get('group_by_aware', True)
            wrap_kwargs = settings.get('wrap_kwargs', None)
            wrap_kwargs = resolve_dict(wrap_kwargs)
            wrap_func = settings.get('wrap_func', None)
            method_kwargs = settings.get('method_kwargs', None)
            method_kwargs = resolve_dict(method_kwargs)
            decorator = settings.get('decorator', None)
            if decorator is None:
                if obj_type in ('red_array', 'records'):
                    decorator = cached_property
                else:
                    decorator = cacheable_property
            decorator_kwargs = merge_dicts(
                dict(obj_type=obj_type, group_by_aware=group_by_aware),
                settings.get('decorator_kwargs', None)
            )
            docstring = settings.get('docstring', None)
            if docstring is None:
                if len(method_kwargs) == 0:
                    docstring = f"`{cls.__name__}.{method_name}` with default arguments."
                else:
                    docstring = f"`{cls.__name__}.{method_name}` with arguments `{method_kwargs}`."

            def new_prop(self,
                         _method_name: str = method_name,
                         _target_name: str = target_name,
                         _use_in_outputs: bool = use_in_outputs,
                         _field_aliases: tp.List[str] = field_aliases,
                         _obj_type: str = obj_type,
                         _group_by_aware: bool = group_by_aware,
                         _wrap_kwargs: tp.Kwargs = wrap_kwargs,
                         _wrap_func: tp.Callable = wrap_func,
                         _method_kwargs: tp.Kwargs = method_kwargs) -> tp.Any:

                def _find_obj(_is_grouped: bool) -> tp.Optional[str]:
                    fields = set(self.in_outputs._fields)
                    for field in _field_aliases:
                        if field in fields:
                            return field
                        if _is_grouped:
                            if _group_by_aware:
                                if field + '_pg' in fields:
                                    return field + '_pg'
                                if field + '_pcg' in fields:
                                    return field + '_pcg'
                                if self.cash_sharing:
                                    if field + '_pcgs' in fields:
                                        return field + '_pcgs'
                            else:
                                if field + '_pc' in fields:
                                    return field + '_pc'
                                if not self.cash_sharing:
                                    if field + '_pcgs' in fields:
                                        return field + '_pcgs'
                        else:
                            if field + '_pc' in fields:
                                return field + '_pc'
                            if field + '_pcg' in fields:
                                return field + '_pcg'
                            if field + '_pcgs' in fields:
                                return field + '_pcgs'
                    return None

                if _use_in_outputs:
                    if self.use_in_outputs and self.in_outputs is not None:
                        is_grouped = self.wrapper.grouper.is_grouped()
                        found_field = _find_obj(is_grouped)
                        if found_field is not None:
                            obj = getattr(self.in_outputs, found_field)
                            if _wrap_func is not None:
                                return _wrap_func(self, obj)
                            if _obj_type == 'array':
                                if _group_by_aware:
                                    return self.wrapper.wrap(obj, **_wrap_kwargs)
                                return self.wrapper.wrap(obj, group_by=False, **_wrap_kwargs)
                            elif _obj_type == 'red_array':
                                _wrap_kwargs = merge_dicts(dict(name_or_index=_target_name), _wrap_kwargs)
                                if _group_by_aware:
                                    return self.wrapper.wrap_reduced(obj, **_wrap_kwargs)
                                return self.wrapper.wrap_reduced(obj, group_by=False, **_wrap_kwargs)
                            else:
                                raise NotImplementedError

                return getattr(self, _method_name)(**_method_kwargs)

            new_prop.__name__ = target_name
            new_prop.__qualname__ = f"{cls.__name__}.{target_name}"
            new_prop.__doc__ = docstring
            setattr(cls, new_prop.__name__, decorator(new_prop, **decorator_kwargs))
        return cls

    return wrapper
