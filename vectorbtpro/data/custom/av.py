# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `AVData`."""

import warnings
from functools import lru_cache
import re
import requests
import urllib.parse

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import (
    split_freq_str,
    prepare_freq,
)
from vectorbtpro.data.custom.remote import RemoteData

__all__ = [
    "AVData",
]

__pdoc__ = {}

AVDataT = tp.TypeVar("AVDataT", bound="AVData")


class AVData(RemoteData):
    """Data class for fetching from Alpha Vantage.

    See https://www.alphavantage.co/documentation/ for API.

    Instead of using https://github.com/RomelTorres/alpha_vantage package, which is stale and has
    many issues, this class parses the API documentation with `AVData.parse_api_meta` using
    `BeautifulSoup4` and builds the API query based on this metadata. It then uses
    [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) to collect
    and format the CSV data.

    This approach is the most flexible we can get since we can instantly react to Alpha Vantage's changes
    in the API. If the data provider changes its API documentation, you can always adapt the parsing
    procedure by overriding `AVData.parse_api_meta`.

    If parser still fails, you can disable parsing entirely and specify all information manually
    by setting `function` and disabling `match_params`

    See `AVData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.AVData.set_custom_settings(
        ...     apikey="YOUR_KEY"
        ... )
        ```

        * Fetch data:

        ```pycon
        >>> data = vbt.AVData.fetch(
        ...     "GOOGL",
        ...     timeframe="1 day",  # premium?
        ... )

        >>> data = vbt.AVData.fetch(
        ...     "BTC_USD",
        ...     timeframe="30 minutes",  # premium?
        ...     category="digital-currency",
        ...     outputsize="full"
        ... )

        >>> data = vbt.AVData.fetch(
        ...     "REAL_GDP",
        ...     category="economic-indicators"
        ... )

        >>> data = vbt.AVData.fetch(
        ...     "IBM",
        ...     category="technical-indicators",
        ...     function="STOCHRSI",
        ...     params=dict(fastkperiod=14)
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.av")

    @classmethod
    def list_symbols(cls, keywords: str, apikey: tp.Optional[str] = None) -> tp.List[str]:
        """List all symbols."""
        av_cfg = cls.get_settings(key_id="custom")

        if apikey is None:
            apikey = av_cfg["apikey"]
        query = dict()
        query["function"] = "SYMBOL_SEARCH"
        query["keywords"] = keywords
        query["datatype"] = "csv"
        query["apikey"] = apikey
        url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(query)
        df = pd.read_csv(url)
        return sorted(df["symbol"].tolist())

    @classmethod
    @lru_cache()
    def parse_api_meta(cls) -> dict:
        """Parse API metadata from the documentation at https://www.alphavantage.co/documentation

        Cached class method. To avoid re-parsing the same metadata in different runtimes, save it manually."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("bs4")

        from bs4 import BeautifulSoup

        page = requests.get("https://www.alphavantage.co/documentation")
        soup = BeautifulSoup(page.content, "html.parser")
        api_meta = {}
        for section in soup.select("article section"):
            category = {}
            function = None
            function_args = dict(req_args=set(), opt_args=set())
            for tag in section.find_all(True):
                if tag.name == "h6":
                    if function is not None and tag.select("b")[0].getText().strip() == "API Parameters":
                        category[function] = function_args
                        function = None
                        function_args = dict(req_args=set(), opt_args=set())
                if tag.name == "b":
                    b_text = tag.getText().strip()
                    if b_text.startswith("❚ Required"):
                        arg = tag.select("code")[0].getText().strip()
                        function_args["req_args"].add(arg)
                if tag.name == "p":
                    p_text = tag.getText().strip()
                    if p_text.startswith("❚ Optional"):
                        arg = tag.select("code")[0].getText().strip()
                        function_args["opt_args"].add(arg)
                if tag.name == "code":
                    code_text = tag.getText().strip()
                    if code_text.startswith("function="):
                        function = code_text.replace("function=", "")
            if function is not None:
                category[function] = function_args
            api_meta[section.select("h2")[0]["id"]] = category

        return api_meta

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        apikey: tp.Optional[str] = None,
        api_meta: tp.Optional[dict] = None,
        category: tp.Optional[str] = None,
        function: tp.Optional[str] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        adjusted: tp.Optional[bool] = None,
        extended: tp.Optional[bool] = None,
        slice: tp.Optional[str] = None,
        series_type: tp.Optional[str] = None,
        time_period: tp.Optional[int] = None,
        outputsize: tp.Optional[str] = None,
        match_params: tp.Optional[bool] = None,
        params: tp.KwargsLike = None,
        read_csv_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Alpha Vantage.

        See https://www.alphavantage.co/documentation/ for API endpoints and their parameters.

        !!! note
            Supports the CSV format only.

        Args:
            symbol (str): Symbol.

                May combine symbol/from_currency and market/to_currency using an underscore.
            apikey (str): API key.
            api_meta (dict): API meta.

                If None, will use `AVData.parse_api_meta` if `function` is not provided
                or `match_params` is True.
            category (str): API category of your choice.

                Used if `function` is not provided or `match_params` is True.

                Supported are:

                * "time-series-data"
                * "fundamentals"
                * "fx"
                * "digital-currency"
                * "economic-indicators"
                * "technical-indicators"
            function (str): API function of your choice.

                If None, will try to resolve it based on other arguments, such as `timeframe`,
                `adjusted`, and `extended`. Required for technical indicators, economic indicators,
                and fundamental data.

                See the keys in sub-dictionaries returned by `AVData.parse_api_meta`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".

                For time series, forex, and crypto, looks for interval type in the function's name.
                Defaults to "60min" if extended, otherwise to "daily".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjusted (bool): Whether to return time series adjusted by historical split and dividend events.
            extended (bool): Whether to return historical intraday time series for the trailing 2 years.
            slice (str): Slice of the trailing 2 years.
            series_type (str): The desired price type in the time series.
            time_period (int): Number of data points used to calculate each window value.
            outputsize (str): Output size.

                Supported are

                * "compact" that returns only the latest 100 data points
                * "full" that returns the full-length time series
            match_params (bool): Whether to match parameters with the ones required by the endpoint.

                Otherwise, uses only (resolved) `function`, `apikey`, `datatype="csv"`, and `params`.
            params: Additional keyword arguments passed as key/value pairs in the URL.
            read_csv_kwargs (dict): Keyword arguments passed to `pd.read_csv`.
            silence_warnings (bool): Whether to silence all warnings.

        For defaults, see `custom.av` in `vectorbtpro._settings.data`.
        """
        av_cfg = cls.get_settings(key_id="custom")

        if apikey is None:
            apikey = av_cfg["apikey"]
        if api_meta is None:
            api_meta = av_cfg["api_meta"]
        if category is None:
            category = av_cfg["category"]
        if function is None:
            function = av_cfg["function"]
        if timeframe is None:
            timeframe = av_cfg["timeframe"]
        if tz is None:
            tz = av_cfg["tz"]
        if adjusted is None:
            adjusted = av_cfg["adjusted"]
        if extended is None:
            extended = av_cfg["extended"]
        if slice is None:
            slice = av_cfg["slice"]
        if series_type is None:
            series_type = av_cfg["series_type"]
        if time_period is None:
            time_period = av_cfg["time_period"]
        if outputsize is None:
            outputsize = av_cfg["outputsize"]
        read_csv_kwargs = merge_dicts(av_cfg["read_csv_kwargs"], read_csv_kwargs)
        if match_params is None:
            match_params = av_cfg["match_params"]
        params = merge_dicts(av_cfg["params"], params)
        if silence_warnings is None:
            silence_warnings = av_cfg["silence_warnings"]

        if api_meta is None and (function is None or match_params):
            if not silence_warnings and cls.parse_api_meta.cache_info().misses == 0:
                warnings.warn("Parsing API documentation...", stacklevel=2)
            try:
                api_meta = cls.parse_api_meta()
            except Exception as e:
                raise ValueError("Can't fetch/parse the API documentation. Specify function and disable match_params.")

        # Resolve the timeframe
        freq = prepare_freq(timeframe)
        interval = None
        interval_type = None
        if timeframe is not None:
            if not isinstance(timeframe, str):
                raise ValueError(f"Invalid timeframe '{timeframe}'")
            split = split_freq_str(timeframe)
            if split is None:
                raise ValueError(f"Invalid timeframe '{timeframe}'")
            multiplier, unit = split
            if unit == "t":
                interval = str(multiplier) + "min"
                interval_type = "INTRADAY"
            elif unit == "h":
                interval = str(60 * multiplier) + "min"
                interval_type = "INTRADAY"
            elif unit == "d":
                interval = "daily"
                interval_type = "DAILY"
            elif unit == "W":
                interval = "weekly"
                interval_type = "WEEKLY"
            elif unit == "M":
                interval = "monthly"
                interval_type = "MONTHLY"
            elif unit == "Q":
                interval = "quarterly"
                interval_type = "QUARTERLY"
            elif unit == "Y":
                interval = "annual"
                interval_type = "ANNUAL"
            if interval is None and multiplier > 1:
                raise ValueError("Multipliers are supported only for intraday timeframes")
        else:
            if extended:
                interval_type = "INTRADAY"
                interval = "60min"
            else:
                interval_type = "DAILY"
                interval = "daily"

        # Resolve the function
        if function is None and category is not None and category == "economic-indicators":
            function = symbol
        if function is None:
            if category is None:
                category = "time-series-data"
            if category in ("technical-indicators", "fundamentals"):
                raise ValueError("Function is required")
            adjusted_in_functions = False
            extended_in_functions = False
            matched_functions = []
            for k, v in api_meta[category].items():
                if interval_type is None or interval_type in k:
                    if "ADJUSTED" in k:
                        adjusted_in_functions = True
                    if "EXTENDED" in k:
                        extended_in_functions = True
                    matched_functions.append(k)

            if adjusted_in_functions:
                matched_functions = [
                    k
                    for k in matched_functions
                    if (adjusted and "ADJUSTED" in k) or (not adjusted and "ADJUSTED" not in k)
                ]
            if extended_in_functions:
                matched_functions = [
                    k
                    for k in matched_functions
                    if (extended and "EXTENDED" in k) or (not extended and "EXTENDED" not in k)
                ]
            if len(matched_functions) == 0:
                raise ValueError("No functions satisfy the requirements")
            if len(matched_functions) > 1:
                raise ValueError("More than one function satisfies the requirements")
            function = matched_functions[0]

        # Resolve the parameters
        if match_params:
            if function is not None and category is None:
                category = None
                for k, v in api_meta.items():
                    if function in v:
                        category = k
                        break
            if category is None:
                raise ValueError("Category is required")
            req_args = api_meta[category][function]["req_args"]
            opt_args = api_meta[category][function]["opt_args"]
            args = set(req_args) | set(opt_args)

            matched_params = dict()
            matched_params["function"] = function
            matched_params["datatype"] = "csv"
            matched_params["apikey"] = apikey
            if "symbol" in args and "market" in args:
                matched_params["symbol"] = symbol.split("_")[0]
                matched_params["market"] = symbol.split("_")[1]
            elif "from_" in args and "to_currency" in args:
                matched_params["from_currency"] = symbol.split("_")[0]
                matched_params["to_currency"] = symbol.split("_")[1]
            elif "from_currency" in args and "to_currency" in args:
                matched_params["from_currency"] = symbol.split("_")[0]
                matched_params["to_currency"] = symbol.split("_")[1]
            elif "symbol" in args:
                matched_params["symbol"] = symbol
            if "interval" in args:
                matched_params["interval"] = interval
            if "adjusted" in args:
                matched_params["adjusted"] = adjusted
            if "extended" in args:
                matched_params["extended"] = extended
            if "slice" in args:
                matched_params["slice"] = slice
            if "series_type" in args:
                matched_params["series_type"] = series_type
            if "time_period" in args:
                matched_params["time_period"] = time_period
            if "outputsize" in args:
                matched_params["outputsize"] = outputsize
            for k, v in params.items():
                if k in args:
                    matched_params[k] = v
                else:
                    raise ValueError(f"Function '{function}' does not expect parameter '{k}'")
            for arg in req_args:
                if arg not in matched_params:
                    raise ValueError(f"Function '{function}' requires parameter '{arg}'")
        else:
            matched_params = dict(params)
            matched_params["function"] = function
            matched_params["apikey"] = apikey
            matched_params["datatype"] = "csv"

        # Collect and format the data
        url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(matched_params)
        df = pd.read_csv(url, **read_csv_kwargs)
        df.index.name = None
        new_columns = []
        for c in df.columns:
            new_c = re.sub(r"^\d+\w*\.\s*", "", c)
            new_c = new_c[0].title() + new_c[1:]
            new_columns.append(new_c)
        df = df.rename(columns=dict(zip(df.columns, new_columns)))
        if not df.empty and df.index[0] > df.index[1]:
            df = df.iloc[::-1]
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is None:
            df = df.tz_localize("UTC")

        return df, dict(tz_convert=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
