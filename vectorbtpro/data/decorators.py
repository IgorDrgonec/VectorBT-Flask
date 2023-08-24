# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Class decorators for data."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import copy_dict

__all__ = []


def attach_symbol_dict_methods(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
    """Class decorator to attach methods for updating symbol dictionaries."""

    checks.assert_subclass_of(cls, "Data")

    for target_name in cls._key_dict_attrs:
        def new_method(self, _target_name=target_name, check_dict_type: bool = True, **kwargs):
            new_kwargs = copy_dict(getattr(self, _target_name))
            for s in self.get_keys(type(new_kwargs)):
                if s not in new_kwargs:
                    new_kwargs[s] = dict()
            for k, v in kwargs.items():
                if check_dict_type:
                    self.check_dict_type(v, k, dict_type=type(new_kwargs))
                if isinstance(v, type(new_kwargs)):
                    for s, _v in v.items():
                        new_kwargs[s][k] = _v
                else:
                    for s in new_kwargs:
                        new_kwargs[s][k] = v
            return self.replace(**{_target_name: new_kwargs})

        new_method.__name__ = "update_" + target_name
        new_method.__doc__ = f"""Update `Data.{target_name}`. Returns a new instance."""
        setattr(cls, new_method.__name__, new_method)

    return cls
