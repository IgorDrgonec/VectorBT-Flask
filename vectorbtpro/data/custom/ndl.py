# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `NDLData`."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import (
    to_timestamp,
    to_tzaware_datetime,
)
from vectorbtpro.data.custom.remote import RemoteData

__all__ = [
    "NDLData",
]

__pdoc__ = {}

NDLDataT = tp.TypeVar("NDLDataT", bound="NDLData")


class NDLData(RemoteData):
    """Data class for fetching from Nasdaq Data Link.

    See https://github.com/Nasdaq/data-link-python for API.

    See `NDLData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.NDLData.set_custom_settings(
        ...     api_key="YOUR_KEY"
        ... )
        ```

        * Fetch data:

        ```pycon
        >>> data = vbt.NDLData.pull(
        ...     "EIA/PET_RWTC_D",
        ...     start="2020-01-01",
        ...     end="2021-01-01"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.ndl")

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        api_key: tp.Optional[str] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        column_indices: tp.Optional[tp.MaybeIterable[int]] = None,
        collapse: tp.Optional[str] = None,
        transform: tp.Optional[str] = None,
        **params,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Nasdaq Data Link.

        Args:
            symbol (str): Symbol.
            api_key (str): API key.
            start (any): Retrieve data rows on and after the specified start date.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): Retrieve data rows up to and including the specified end date.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            column_indices (int or iterable): Request one or more specific columns.

                Column 0 is the date column and is always returned. Data begins at column 1.
            collapse (str): Change the sampling frequency of the returned data.

                Options are "daily", "weekly", "monthly", "quarterly", and "annual".
            transform (str): Perform elementary calculations on the data prior to downloading.

                Options are "diff", "rdiff", "cumul", and "normalize".
            **params: Keyword arguments sent as field/value params to Nasdaq Data Link with no interference.

        For defaults, see `custom.ndl` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("nasdaqdatalink")

        import nasdaqdatalink

        ndl_cfg = cls.get_settings(key_id="custom")

        if api_key is None:
            api_key = ndl_cfg["api_key"]
        if start is None:
            start = ndl_cfg["start"]
        if end is None:
            end = ndl_cfg["end"]
        if tz is None:
            tz = ndl_cfg["tz"]
        if column_indices is None:
            column_indices = ndl_cfg["column_indices"]
        if column_indices is not None:
            if isinstance(column_indices, int):
                dataset = symbol + "." + str(column_indices)
            else:
                dataset = [symbol + "." + str(index) for index in column_indices]
        else:
            dataset = symbol
        if collapse is None:
            collapse = ndl_cfg["collapse"]
        if transform is None:
            transform = ndl_cfg["transform"]
        params = merge_dicts(ndl_cfg["params"], params)

        # Establish the timestamps
        if start is not None:
            start = to_tzaware_datetime(start, naive_tz=tz, tz="UTC")
            start_date = pd.Timestamp(start).isoformat()
        else:
            start_date = None
        if end is not None:
            end = to_tzaware_datetime(end, naive_tz=tz, tz="UTC")
            end_date = pd.Timestamp(end).isoformat()
        else:
            end_date = None

        # Collect and format the data
        df = nasdaqdatalink.get(
            dataset,
            api_key=api_key,
            start_date=start_date,
            end_date=end_date,
            collapse=collapse,
            transform=transform,
            **params,
        )
        new_columns = []
        for c in df.columns:
            new_c = c
            if isinstance(symbol, str):
                new_c = new_c.replace(symbol + " - ", "")
            if new_c == "Last":
                new_c = "Close"
            new_columns.append(new_c)
        df = df.rename(columns=dict(zip(df.columns, new_columns)))

        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is None:
            df = df.tz_localize("UTC")
        if not df.empty:
            if start is not None:
                start = to_timestamp(start, tz=df.index.tzinfo)
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                end = to_timestamp(end, tz=df.index.tzinfo)
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz_convert=tz)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
