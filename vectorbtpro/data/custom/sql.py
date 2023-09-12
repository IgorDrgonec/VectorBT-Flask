# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `SQLData`."""

from typing import Iterator

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.db import DBData
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import to_tzaware_datetime, to_naive_datetime

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from sqlalchemy import Engine as EngineT, Selectable as SelectableT
except ImportError:
    EngineT = tp.Any

__all__ = [
    "SQLData",
]

__pdoc__ = {}


class SQLData(DBData):
    """Data class for fetching data from a database using SQLAlchemy."""

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.sql")

    @classmethod
    def resolve_engine(cls,
        engine: tp.Union[None, str, EngineT] = None,
        return_meta: bool = False,
        **engine_config,
    ) -> tp.Union[EngineT, dict]:
        """Resolve the engine.

        If provided, must be of the type `sqlalchemy.engine.base.Engine`.
        Otherwise, will be created using `engine_config`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import create_engine

        cfg = cls.get_settings(key_id="custom")

        engine = cls.resolve_argument(engine, "engine", cfg=cfg)
        if engine is None:
            raise ValueError("Must provide engine or URL (via engine argument)")
        if isinstance(engine, str):
            engine_name = engine
        else:
            engine_name = None
        if engine_name is not None:
            if "engine" in cfg["engines"].get(engine_name, {}):
                engine = cfg["engines"][engine_name]["engine"]

        has_engine_config = len(engine_config) > 0
        engine_config = cls.resolve_argument(
            engine_config, "engine_config", is_dict=True, engine_name=engine_name, cfg=cfg
        )
        if isinstance(engine, str):
            engine = create_engine(engine, **engine_config)
            should_dispose = True
        else:
            if has_engine_config:
                raise ValueError("Cannot apply engine_config on already created engine")
            should_dispose = False
        if return_meta:
            return dict(
                engine=engine,
                engine_name=engine_name,
                should_dispose=should_dispose,
            )
        return engine

    @classmethod
    def list_schemas(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        engine: tp.Union[None, str, EngineT] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.List[str]:
        """List all schemas.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each symbol against `pattern`.

        Keyword arguments `**kwargs` are passed to `inspector.get_schema_names`.

        If `dispose_engine` is None, disposes the engine if it wasn't provided."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        inspector = inspect(engine)
        all_schemas = []
        for schema in inspector.get_schema_names(**kwargs):
            if pattern is not None:
                if not cls.key_match(schema, pattern, use_regex=use_regex):
                    continue
            all_schemas.append(schema)
        if dispose_engine:
            engine.dispose()
        return sorted(all_schemas)

    @classmethod
    def list_tables(
        cls,
        *,
        schema_pattern: tp.Optional[str] = None,
        table_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        schema: tp.Optional[str] = None,
        incl_views: bool = True,
        engine: tp.Union[None, str, EngineT] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.List[str]:
        """List all tables and views.

        If `schema` is None, searches for all schema names in the database and prefixes each table
        with the respective schema name (unless there's only one schema "main"). If `schema` is False,
        sets the schema to None. If `schema` is provided, returns the tables corresponding to this
        schema without a prefix.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each schema against
        `schema_pattern` and each table against `table_pattern`.

        Keyword arguments `**kwargs` are passed to `inspector.get_table_names`.

        If `dispose_engine` is None, disposes the engine if it wasn't provided."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        if schema is None:
            schemas = cls.list_schemas(
                pattern=schema_pattern,
                use_regex=use_regex,
                engine=engine,
                **kwargs,
            )
            if len(schemas) == 0:
                schemas = [None]
                prefix_schema = False
            elif len(schemas) == 1 and schemas[0] == "main":
                prefix_schema = False
            else:
                prefix_schema = True
        elif schema is False:
            schemas = [None]
            prefix_schema = False
        else:
            schemas = [schema]
            prefix_schema = False
        inspector = inspect(engine)
        all_tables = []
        for schema in schemas:
            table_names = inspector.get_table_names(schema, **kwargs)
            if incl_views:
                try:
                    table_names += inspector.get_view_names(schema, **kwargs)
                except NotImplementedError as e:
                    pass
                try:
                    table_names += inspector.get_materialized_view_names(schema, **kwargs)
                except NotImplementedError as e:
                    pass
            for table in table_names:
                if table_pattern is not None:
                    if not cls.key_match(table, table_pattern, use_regex=use_regex):
                        continue
                if prefix_schema and schema is not None:
                    all_tables.append(str(schema) + ":" + table)
                else:
                    all_tables.append(table)
        if dispose_engine:
            engine.dispose()
        return sorted(all_tables)

    @classmethod
    def resolve_argument(
        cls,
        arg_value: tp.Any,
        arg_name: str,
        is_dict: bool = False,
        engine_name: tp.Optional[str] = None,
        cfg: tp.Optional[dict] = None,
    ) -> tp.Any:
        """Resolve an argument with respect to global settings."""
        if cfg is None:
            cfg = cls.get_settings(key_id="custom")
        if is_dict:
            if engine_name is not None:
                return merge_dicts(
                    cfg[arg_name],
                    cfg["engines"].get(engine_name, {}).get(arg_name, {}),
                    arg_value,
                )
            return merge_dicts(cfg[arg_name], arg_value)
        if arg_value is not None:
            return arg_value
        if engine_name is not None:
            return cfg["engines"].get(engine_name, {}).get(arg_name, cfg[arg_name])
        return cfg[arg_name]

    @classmethod
    def fetch_table(
        cls,
        table_name: str,
        schema: tp.Optional[str] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        to_utc: tp.Union[None, bool, str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        index_col: tp.Union[None, bool, tp.MaybeList[tp.IntStr]] = None,
        columns: tp.Optional[tp.MaybeList[tp.IntStr]] = None,
        parse_dates: tp.Union[None, tp.List[tp.IntStr], tp.Dict[tp.IntStr, tp.Any]] = None,
        dtype: tp.Union[None, tp.DTypeLike, tp.Dict[tp.IntStr, tp.DTypeLike]] = None,
        chunksize: tp.Optional[int] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        squeeze: tp.Optional[bool] = None,
        **read_sql_kwargs,
    ) -> tp.KeyData:
        """Fetch the table of a feature or symbol."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import MetaData, and_

        cfg = cls.get_settings(key_id="custom")

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        engine_name = engine_meta["engine_name"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        if ":" in table_name:
            schema, table_name = table_name.split(":")
        if schema is None:
            schema = cfg["schema"]
        metadata_obj = MetaData()
        metadata_obj.reflect(bind=engine, schema=schema, only=[table_name], views=True)
        table = metadata_obj.tables[table_name]
        table_column_names = [c.name for c in table.columns]

        start = cls.resolve_argument(start, "start", engine_name=engine_name, cfg=cfg)
        end = cls.resolve_argument(end, "end", engine_name=engine_name, cfg=cfg)
        to_utc = cls.resolve_argument(to_utc, "to_utc", engine_name=engine_name, cfg=cfg)
        tz = cls.resolve_argument(tz, "tz", engine_name=engine_name, cfg=cfg)
        index_col = cls.resolve_argument(index_col, "index_col", engine_name=engine_name, cfg=cfg)
        if index_col is False:
            index_col = None
        columns = cls.resolve_argument(columns, "columns", engine_name=engine_name, cfg=cfg)
        parse_dates = cls.resolve_argument(parse_dates, "parse_dates", engine_name=engine_name, cfg=cfg)
        dtype = cls.resolve_argument(dtype, "dtype", engine_name=engine_name, cfg=cfg)
        chunksize = cls.resolve_argument(chunksize, "chunksize", engine_name=engine_name, cfg=cfg)
        chunk_func = cls.resolve_argument(chunk_func, "chunk_func", engine_name=engine_name, cfg=cfg)
        squeeze = cls.resolve_argument(squeeze, "squeeze", engine_name=engine_name, cfg=cfg)
        read_sql_kwargs = cls.resolve_argument(
            read_sql_kwargs, "read_sql_kwargs", is_dict=True, engine_name=engine_name, cfg=cfg
        )

        def _resolve_columns(c):
            if checks.is_int(c):
                c = table_column_names[int(c)]
            elif not isinstance(c, str):
                new_c = []
                for _c in c:
                    if checks.is_int(_c):
                        new_c.append(table_column_names[int(_c)])
                    else:
                        if _c not in table_column_names:
                            for __c in table_column_names:
                                if _c.lower() == __c.lower():
                                    _c = __c
                                    break
                        new_c.append(_c)
                c = new_c
            else:
                if c not in table_column_names:
                    for _c in table_column_names:
                        if c.lower() == _c.lower():
                            return _c
            return c

        if index_col is not None:
            index_col = _resolve_columns(index_col)
            if isinstance(index_col, str):
                index_col = [index_col]
        if columns is not None:
            columns = _resolve_columns(columns)
            if isinstance(columns, str):
                columns = [columns]
        if parse_dates is not None:
            if not isinstance(parse_dates, dict):
                parse_dates = _resolve_columns(parse_dates)
            else:
                parse_dates = dict(zip(_resolve_columns(parse_dates.keys()), parse_dates.values()))
        if dtype is not None:
            if isinstance(dtype, dict):
                dtype = dict(zip(_resolve_columns(dtype.keys()), dtype.values()))

        selection = table.select()
        if index_col is not None and columns is not None:
            columns = index_col + columns
        if columns is not None:
            selection = selection.with_only_columns(*[table.c.get(c) for c in columns])
        if start is not None or end is not None:
            if index_col is None:
                raise ValueError("Must provide index column for filtering by index")
            first_obj = pd.read_sql(
                selection.limit(1),
                engine,
                index_col=index_col,
                parse_dates=parse_dates,
                dtype=dtype,
                chunksize=None,
                **read_sql_kwargs,
            )
            if isinstance(first_obj.index, pd.DatetimeIndex):
                if tz is None:
                    tz = first_obj.index.tz
                if first_obj.index.tz is not None:
                    if start is not None:
                        start = to_tzaware_datetime(start, naive_tz=tz, tz=first_obj.index.tz)
                    if end is not None:
                        end = to_tzaware_datetime(end, naive_tz=tz, tz=first_obj.index.tz)
                else:
                    if start is not None:
                        if to_utc is True or (isinstance(to_utc, str) and to_utc.lower() == "index"):
                            start = to_tzaware_datetime(start, naive_tz=tz, tz="utc")
                            start = to_naive_datetime(start)
                        else:
                            start = to_naive_datetime(start, tz=tz)
                    if end is not None:
                        if to_utc is True or (isinstance(to_utc, str) and to_utc.lower() == "index"):
                            end = to_tzaware_datetime(end, naive_tz=tz, tz="utc")
                            end = to_naive_datetime(end)
                        else:
                            end = to_naive_datetime(end, tz=tz)

            def _to_native_type(x):
                if checks.is_np_scalar(x):
                    return x.item()
                return x

            and_list = []
            if start is not None:
                if len(index_col) > 1:
                    if not isinstance(start, tuple):
                        raise TypeError("Start must be a tuple if the index is a multi-index")
                    if len(start) != len(index_col):
                        raise ValueError("Start tuple must match the number of levels in the multi-index")
                    for i in range(len(index_col)):
                        index_column = table.c.get(index_col[i])
                        and_list.append(index_column >= _to_native_type(start[i]))
                else:
                    index_column = table.c.get(index_col[0])
                    and_list.append(index_column >= _to_native_type(start))
            if end is not None:
                if len(index_col) > 1:
                    if not isinstance(end, tuple):
                        raise TypeError("End must be a tuple if the index is a multi-index")
                    if len(end) != len(index_col):
                        raise ValueError("End tuple must match the number of levels in the multi-index")
                    for i in range(len(index_col)):
                        index_column = table.c.get(index_col[i])
                        and_list.append(index_column < _to_native_type(end[i]))
                else:
                    index_column = table.c.get(index_col[0])
                    and_list.append(index_column < _to_native_type(end))
            selection = selection.where(and_(*and_list))

        obj = pd.read_sql(
            selection,
            engine,
            index_col=index_col,
            parse_dates=parse_dates,
            dtype=dtype,
            chunksize=chunksize,
            **read_sql_kwargs,
        )
        if isinstance(obj, Iterator):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if to_utc is not False:
            if to_utc is True or (isinstance(to_utc, str) and to_utc.lower() == "index"):
                if isinstance(obj.index, pd.DatetimeIndex):
                    obj = obj.copy(deep=False)
                    if obj.index.tz is None:
                        obj.index = obj.index.tz_localize("utc")
            elif to_utc is True or (isinstance(to_utc, str) and to_utc.lower() == "columns"):
                if isinstance(obj, pd.Series):
                    if hasattr(obj, "dt"):
                        if obj.dt.tz is None:
                            obj = obj.dt.tz_localize("utc")
                else:
                    has_dt_column = False
                    for c in range(len(obj.columns)):
                        if hasattr(obj[c], "dt"):
                            has_dt_column = True
                            break
                    if has_dt_column:
                        obj = obj.copy(deep=False)
                        for c in range(len(obj.columns)):
                            if hasattr(obj[c], "dt"):
                                if obj.dt.tz is None:
                                    obj[c] = obj[c].dt.tz_localize("utc")
        if dispose_engine:
            engine.dispose()
        return obj, dict(tz_convert=tz)

    @classmethod
    def fetch_feature(cls, feature: str, **kwargs) -> tp.FeatureData:
        """Fetch the table of a feature.

        Uses `SQLData.fetch_key`."""
        return cls.fetch_table(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: str, **kwargs) -> tp.SymbolData:
        """Fetch the table for a symbol.

        Uses `SQLData.fetch_key`."""
        return cls.fetch_table(symbol, **kwargs)

    def update_key(self, key: str, **kwargs) -> tp.KeyData:
        """Update data of a feature or symbol."""
        fetch_kwargs = self.select_fetch_kwargs(key)
        fetch_kwargs["start"] = self.select_last_index(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if self.feature_oriented:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: str, **kwargs) -> tp.FeatureData:
        """Update data of a feature.

        Uses `SQLData.update_key`."""
        return self.update_key(feature, **kwargs)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        """Update data for a symbol.

        Uses `SQLData.update_key`."""
        return self.update_key(symbol, **kwargs)
