# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `RemoteData`."""

import time
import traceback
import warnings
from functools import wraps, partial
import requests

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import (
    to_tzaware_datetime,
    datetime_to_ms,
    split_freq_str,
    prepare_freq,
)
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.data.custom.remote import RemoteData

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from polygon import RESTClient as PolygonClientT
except ImportError:
    PolygonClientT = tp.Any

__all__ = [
    "RemoteData",
]

__pdoc__ = {}

PolygonDataT = tp.TypeVar("PolygonDataT", bound="PolygonData")


class PolygonData(RemoteData):
    """Data class for fetching from Polygon.

    See https://github.com/polygon-io/client-python for API.

    See `PolygonData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally:

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.PolygonData.set_custom_settings(
        ...     client_config=dict(
        ...         api_key="YOUR_KEY"
        ...     )
        ... )
        ```

        * Fetch stock data:

        ```pycon
        >>> data = vbt.PolygonData.pull(
        ...     "AAPL",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```

        * Fetch crypto data:

        ```pycon
        >>> data = vbt.PolygonData.pull(
        ...     "X:BTCUSD",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.polygon")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        client: tp.Optional[PolygonClientT] = None,
        client_config: tp.DictLike = None,
        **list_tickers_kwargs,
    ) -> tp.List[str]:
        """List all symbols.

        Uses `CustomData.key_match` to check each symbol against `pattern`.

        For supported keyword arguments, see `polygon.RESTClient.list_tickers`."""
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        all_symbols = []
        for ticker in client.list_tickers(**list_tickers_kwargs):
            symbol = ticker.ticker
            if pattern is not None:
                if not cls.key_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)
        return sorted(all_symbols)

    @classmethod
    def resolve_client(cls, client: tp.Optional[PolygonClientT] = None, **client_config) -> PolygonClientT:
        """Resolve the client.

        If provided, must be of the type `polygon.rest.RESTClient`.
        Otherwise, will be created using `client_config`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("polygon")
        from polygon import RESTClient

        polygon_cfg = cls.get_settings(key_id="custom")

        if client is None:
            client = polygon_cfg["client"]
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = merge_dicts(polygon_cfg["client_config"], client_config)
        if client is None:
            client = RESTClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[PolygonClientT] = None,
        client_config: tp.DictLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        adjusted: tp.Optional[bool] = None,
        limit: tp.Optional[int] = None,
        params: tp.KwargsLike = None,
        delay: tp.Optional[float] = None,
        retries: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Polygon.

        Args:
            symbol (str): Symbol.

                Supports the following APIs:

                * Stocks and equities
                * Currencies - symbol must have the prefix `C:`
                * Crypto - symbol must have the prefix `X:`
            client (polygon.rest.RESTClient): Client.

                See `PolygonData.resolve_client`.
            client_config (dict): Client config.

                See `PolygonData.resolve_client`.
            start (any): The start of the aggregate time window.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): The end of the aggregate time window.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjusted (str): Whether the results are adjusted for splits.

                By default, results are adjusted.
                Set this to False to get results that are NOT adjusted for splits.
            limit (int): Limits the number of base aggregates queried to create the aggregate results.

                Max 50000 and Default 5000.
            params (dict): Any additional query params.
            delay (float): Time to sleep after each request (in milliseconds).
            retries (int): The number of retries on failure to fetch data.
            show_progress (bool): Whether to show the progress bar.
            pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.
            silence_warnings (bool): Whether to silence all warnings.

        For defaults, see `custom.polygon` in `vectorbtpro._settings.data`.

        !!! note
            If you're using a free plan that has an API rate limit of several requests per minute,
            make sure to set `delay` to a higher number, such as 12000 (which makes 5 requests per minute).
        """
        polygon_cfg = cls.get_settings(key_id="custom")

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        if start is None:
            start = polygon_cfg["start"]
        if end is None:
            end = polygon_cfg["end"]
        if timeframe is None:
            timeframe = polygon_cfg["timeframe"]
        if tz is None:
            tz = polygon_cfg["tz"]
        if adjusted is None:
            adjusted = polygon_cfg["adjusted"]
        if limit is None:
            limit = polygon_cfg["limit"]
        params = merge_dicts(polygon_cfg["params"], params)
        if delay is None:
            delay = polygon_cfg["delay"]
        if retries is None:
            retries = polygon_cfg["retries"]
        if show_progress is None:
            show_progress = polygon_cfg["show_progress"]
        pbar_kwargs = merge_dicts(polygon_cfg["pbar_kwargs"], pbar_kwargs)
        if silence_warnings is None:
            silence_warnings = polygon_cfg["silence_warnings"]

        # Resolve the timeframe
        freq = prepare_freq(timeframe)
        if not isinstance(timeframe, str):
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        split = split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        multiplier, unit = split
        if unit == "t":
            unit = "minute"
        elif unit == "h":
            unit = "hour"
        elif unit == "d":
            unit = "day"
        elif unit == "W":
            unit = "week"
        elif unit == "M":
            unit = "month"
        elif unit == "Q":
            unit = "quarter"
        elif unit == "Y":
            unit = "year"

        # Establish the timestamps
        if start is not None:
            start_ts = datetime_to_ms(to_tzaware_datetime(start, naive_tz=tz, tz="UTC"))
        else:
            start_ts = None
        if end is not None:
            end_ts = datetime_to_ms(to_tzaware_datetime(end, naive_tz=tz, tz="UTC"))
        else:
            end_ts = None
        prev_end_ts = None

        def _retry(method):
            @wraps(method)
            def retry_method(*args, **kwargs):
                for i in range(retries):
                    try:
                        return method(*args, **kwargs)
                    except requests.exceptions.HTTPError as e:
                        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 429:
                            if not silence_warnings:
                                warnings.warn(traceback.format_exc(), stacklevel=2)
                                # Polygon.io API rate limit is per minute
                                warnings.warn("Waiting 1 minute...", stacklevel=2)
                            time.sleep(60)
                        else:
                            raise e
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        if i == retries - 1:
                            raise e
                        if not silence_warnings:
                            warnings.warn(traceback.format_exc(), stacklevel=2)
                        if delay is not None:
                            time.sleep(delay / 1000)

            return retry_method

        def _postprocess(agg):
            return dict(
                o=agg.open,
                h=agg.high,
                l=agg.low,
                c=agg.close,
                v=agg.volume,
                vw=agg.vwap,
                t=agg.timestamp,
                n=agg.transactions,
            )

        @_retry
        def _fetch(_start_ts, _limit):
            return list(
                map(
                    _postprocess,
                    client.get_aggs(
                        ticker=symbol,
                        multiplier=multiplier,
                        timespan=unit,
                        from_=_start_ts,
                        to=end_ts,
                        adjusted=adjusted,
                        sort="asc",
                        limit=_limit,
                        params=params,
                        raw=False,
                    ),
                )
            )

        def _ts_to_str(ts: tp.Optional[int]) -> str:
            if ts is None:
                return "/"
            return str(pd.Timestamp(ts, unit="ms", tz="utc"))

        def _filter_func(d: tp.Dict, _prev_end_ts: tp.Optional[int] = None) -> bool:
            if start_ts is not None:
                if d["t"] < start_ts:
                    return False
            if _prev_end_ts is not None:
                if d["t"] <= _prev_end_ts:
                    return False
            if end_ts is not None:
                if d["t"] >= end_ts:
                    return False
            return True

        # Iteratively collect the data
        data = []
        try:
            with get_pbar(show_progress=show_progress, **pbar_kwargs) as pbar:
                pbar.set_description(_ts_to_str(start_ts if prev_end_ts is None else prev_end_ts))
                while True:
                    # Fetch the klines for the next timeframe
                    next_data = _fetch(start_ts if prev_end_ts is None else prev_end_ts, limit)
                    next_data = list(filter(partial(_filter_func, _prev_end_ts=prev_end_ts), next_data))

                    # Update the timestamps and the progress bar
                    if not len(next_data):
                        break
                    data += next_data
                    if start_ts is None:
                        start_ts = next_data[0]["t"]
                    pbar.set_description(
                        "{} - {}".format(
                            _ts_to_str(start_ts),
                            _ts_to_str(next_data[-1]["t"]),
                        )
                    )
                    pbar.update(1)
                    prev_end_ts = next_data[-1]["t"]
                    if end_ts is not None and prev_end_ts >= end_ts:
                        break
                    if delay is not None:
                        time.sleep(delay / 1000)  # be kind to api
        except Exception as e:
            if not silence_warnings:
                warnings.warn(traceback.format_exc(), stacklevel=2)
                warnings.warn(
                    (
                        f"Symbol '{str(symbol)}' raised an exception. Returning incomplete data. "
                        "Use update() method to fetch missing data."
                    ),
                    stacklevel=2,
                )

        df = pd.DataFrame(data)
        df = df[["t", "o", "h", "l", "c", "v", "n", "vw"]]
        df = df.rename(
            columns={
                "t": "Open time",
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
                "n": "Trade count",
                "vw": "VWAP",
            }
        )
        df.index = pd.to_datetime(df["Open time"], unit="ms", utc=True)
        del df["Open time"]
        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        if "Trade count" in df.columns:
            df["Trade count"] = df["Trade count"].astype(int, errors="ignore")
        if "VWAP" in df.columns:
            df["VWAP"] = df["VWAP"].astype(float)

        return df, dict(tz_convert=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        fetch_kwargs["start"] = self.last_index[symbol]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


PolygonData.override_feature_config_doc(__pdoc__)
