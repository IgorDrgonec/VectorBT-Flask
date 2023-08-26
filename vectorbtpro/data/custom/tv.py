# Copyright (c) 2021-2023 Oleg Polakow. All rights reserved.

"""Module with `TVData`."""

import datetime
import random
import re
import string
import pandas as pd
import requests
import json
import time
from websocket import WebSocket

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts, Configured
from vectorbtpro.utils.pbar import get_pbar
from vectorbtpro.utils.datetime_ import (
    split_freq_str,
    prepare_freq,
)
from vectorbtpro.data.custom.remote import RemoteData

__all__ = [
    "TVClient",
    "TVData",
]

__pdoc__ = {}

SIGNIN_URL = "https://www.tradingview.com/accounts/signin/"
SEARCH_URL = "https://symbol-search.tradingview.com/symbol_search/v3/?text={}&exchange={}&start={}&hl=2&lang=en&domain=production"
SCAN_URL = "https://scanner.tradingview.com/{}/scan"
ORIGIN_URL = "https://data.tradingview.com"
REFERER_URL = "https://www.tradingview.com"
WS_URL = "wss://data.tradingview.com/socket.io/websocket"
PRO_WS_URL = "wss://prodata.tradingview.com/socket.io/websocket"
WS_TIMEOUT = 5


class TVClient(Configured):
    """Client for TradingView."""

    _expected_keys: tp.ClassVar[tp.Optional[tp.Set[str]]] = (Configured._expected_keys or set()) | {
        "username",
        "password",
        "auth_token",
        "user_agent",
    }

    def __init__(
        self,
        username: tp.Optional[str] = None,
        password: tp.Optional[str] = None,
        auth_token: tp.Optional[str] = None,
        user_agent: tp.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Client for TradingView."""
        Configured.__init__(
            self,
            username=username,
            password=password,
            auth_token=auth_token,
            user_agent=user_agent,
            **kwargs,
        )

        if auth_token is None:
            auth_token = self.auth(username, password, user_agent=user_agent)
        elif username is not None or password is not None:
            raise ValueError("Either username and password, or auth_token must be provided")

        self._auth_token = auth_token
        self._ws = None
        self._session = self.generate_session()
        self._chart_session = self.generate_chart_session()

    @property
    def auth_token(self) -> str:
        """Authentication token."""
        return self._auth_token

    @property
    def ws(self) -> WebSocket:
        """Instance of `websocket.Websocket`."""
        return self._ws

    @property
    def session(self) -> str:
        """Session."""
        return self._session

    @property
    def chart_session(self) -> str:
        """Chart session."""
        return self._chart_session

    @staticmethod
    def auth(
        username: tp.Optional[str] = None,
        password: tp.Optional[str] = None,
        user_agent: tp.Optional[str] = None,
    ) -> str:
        """Authenticate."""
        if username is not None and password is not None:
            data = {"username": username, "password": password, "remember": "on"}
            headers = {"Referer": REFERER_URL}
            if user_agent is not None:
                headers["User-Agent"] = user_agent
            response = requests.post(url=SIGNIN_URL, data=data, headers=headers)
            response.raise_for_status()
            json = response.json()
            if "user" not in json or "auth_token" not in json["user"]:
                raise ValueError(json)
            return json["user"]["auth_token"]
        if username is not None or password is not None:
            raise ValueError("Both username and password must be provided")
        return "unauthorized_user_token"

    @staticmethod
    def generate_session() -> str:
        """Generate session."""
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(stringLength))
        return "qs_" + random_string

    @staticmethod
    def generate_chart_session() -> str:
        """Generate chart session."""
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(stringLength))
        return "cs_" + random_string

    def create_connection(self, pro_data: bool = True) -> None:
        """Create a websocket connection."""
        from websocket import create_connection

        if pro_data:
            self._ws = create_connection(PRO_WS_URL, headers=json.dumps({"Origin": ORIGIN_URL}), timeout=WS_TIMEOUT)
        else:
            self._ws = create_connection(WS_URL, headers=json.dumps({"Origin": ORIGIN_URL}), timeout=WS_TIMEOUT)

    @staticmethod
    def filter_raw_message(text) -> tp.Tuple[str, str]:
        """Filter raw message."""
        found = re.search('"m":"(.+?)",', text).group(1)
        found2 = re.search('"p":(.+?"}"])}', text).group(1)
        return found, found2

    @staticmethod
    def prepend_header(st: str) -> str:
        """Prepend a header."""
        return "~m~" + str(len(st)) + "~m~" + st

    @staticmethod
    def construct_message(func: str, param_list: tp.List[str]) -> str:
        """Construct a message."""
        return json.dumps({"m": func, "p": param_list}, separators=(",", ":"))

    def create_message(self, func: str, param_list: tp.List[str]) -> str:
        """Create a message."""
        return self.prepend_header(self.construct_message(func, param_list))

    def send_message(self, func: str, param_list: tp.List[str]) -> None:
        """Send a message."""
        m = self.create_message(func, param_list)
        self.ws.send(m)

    @staticmethod
    def convert_raw_data(raw_data: str, symbol: str) -> pd.DataFrame:
        """Process raw data into a DataFrame."""
        search_result = re.search(r'"s":\[(.+?)\}\]', raw_data)
        if search_result is None:
            raise ValueError("Couldn't parse data returned by TradingView")
        out = search_result.group(1)
        x = out.split(',{"')
        data = list()
        volume_data = True
        for xi in x:
            xi = re.split(r"\[|:|,|\]", xi)
            ts = datetime.datetime.utcfromtimestamp(float(xi[4]))
            row = [ts]
            for i in range(5, 10):
                # skip converting volume data if does not exists
                if not volume_data and i == 9:
                    row.append(0.0)
                    continue
                try:
                    row.append(float(xi[i]))
                except ValueError:
                    volume_data = False
                    row.append(0.0)
            data.append(row)
        data = pd.DataFrame(data, columns=["datetime", "open", "high", "low", "close", "volume"])
        data = data.set_index("datetime")
        data.insert(0, "symbol", value=symbol)
        return data

    @staticmethod
    def format_symbol(symbol: str, exchange: str, fut_contract: tp.Optional[int] = None) -> str:
        """Format a symbol."""
        if ":" in symbol:
            pass
        elif fut_contract is None:
            symbol = f"{exchange}:{symbol}"
        elif isinstance(fut_contract, int):
            symbol = f"{exchange}:{symbol}{fut_contract}!"
        else:
            raise ValueError(f"Invalid option fut_contract='{fut_contract}'")
        return symbol

    def get_hist(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1D",
        fut_contract: tp.Optional[int] = None,
        adjustment: str = "splits",
        extended_session: bool = False,
        pro_data: bool = True,
        limit: int = 20000,
        return_raw: bool = False,
    ) -> tp.Union[str, tp.Frame]:
        """Get historical data."""
        symbol = self.format_symbol(symbol=symbol, exchange=exchange, fut_contract=fut_contract)

        self.create_connection(pro_data=pro_data)
        self.send_message("set_auth_token", [self.auth_token])
        self.send_message("chart_create_session", [self.chart_session, ""])
        self.send_message("quote_create_session", [self.session])
        self.send_message(
            "quote_set_fields",
            [
                self.session,
                "ch",
                "chp",
                "current_session",
                "description",
                "local_description",
                "language",
                "exchange",
                "fractional",
                "is_tradable",
                "lp",
                "lp_time",
                "minmov",
                "minmove2",
                "original_name",
                "pricescale",
                "pro_name",
                "short_name",
                "type",
                "update_mode",
                "volume",
                "currency_code",
                "rchp",
                "rtc",
            ],
        )
        self.send_message("quote_add_symbols", [self.session, symbol, {"flags": ["force_permission"]}])
        self.send_message("quote_fast_symbols", [self.session, symbol])
        self.send_message(
            "resolve_symbol",
            [
                self.chart_session,
                "symbol_1",
                '={"symbol":"'
                + symbol
                + '","adjustment":"'
                + adjustment
                + '","session":'
                + ('"regular"' if not extended_session else '"extended"')
                + "}",
            ],
        )
        self.send_message("create_series", [self.chart_session, "s1", "s1", "symbol_1", interval, limit])
        self.send_message("switch_timezone", [self.chart_session, "exchange"])

        raw_data = ""
        while True:
            try:
                result = self.ws.recv()
                raw_data = raw_data + result + "\n"
            except Exception as e:
                break
            if "series_completed" in result:
                break
        if return_raw:
            return raw_data
        return self.convert_raw_data(raw_data, symbol)

    @staticmethod
    def search_symbol(
        text: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        delay: tp.Optional[int] = None,
        show_progress: bool = True,
        pbar_kwargs: tp.KwargsLike = None,
    ) -> tp.List[dict]:
        """Search for a symbol."""
        if text is None:
            text = ""
        if exchange is None:
            exchange = ""
        if pbar_kwargs is None:
            pbar_kwargs = {}
        symbols_remaining = None
        symbols_list = []
        pbar = None

        while symbols_remaining is None or symbols_remaining > 0:
            url = SEARCH_URL.format(text, exchange.upper(), len(symbols_list))
            resp = requests.get(url)
            symbols_data = json.loads(resp.text)
            symbols_remaining = symbols_data.get("symbols_remaining", 0)
            new_symbols = symbols_data.get("symbols", [])
            symbols_list.extend(new_symbols)
            if pbar is None and symbols_remaining > 0:
                pbar = get_pbar(
                    total=len(new_symbols) + symbols_remaining,
                    show_progress=show_progress,
                    **pbar_kwargs,
                )
            if pbar is not None:
                pbar.update(len(new_symbols))
            if delay is not None:
                time.sleep(delay / 1000)
        if pbar is not None:
            pbar.close()
        return symbols_list

    @staticmethod
    def scan_symbols(market: str) -> tp.List[dict]:
        """Scan symbols in a region/market."""
        url = SCAN_URL.format(market.lower())
        resp = requests.get(url)
        symbols_list = json.loads(resp.text)["data"]
        return symbols_list


TVDataT = tp.TypeVar("TVDataT", bound="TVData")


class TVData(RemoteData):
    """Data class for fetching from TradingView.

    See `TVData.fetch_symbol` for arguments.

    !!! note
        If you're getting the error "Please confirm that you are not a robot by clicking the captcha box."
        when attempting to authenticate, use `auth_token` instead of `username` and `password`.
        To get the authentication token, go to TradingView, log in, visit any chart, open your console's
        developer tools, and search for "auth_token".

    Usage:
        * Set up the credentials globally (optional):

        ```pycon
        >>> import vectorbtpro as vbt

        >>> vbt.TVData.set_custom_settings(
        ...     client_config=dict(
        ...         username="YOUR_USERNAME",
        ...         password="YOUR_PASSWORD",
        ...         auth_token="YOUR_AUTH_TOKEN",  # optional, instead of username and password
        ...         user_agent="YOUR_USER_AGENT"  # optional, see https://useragentstring.com/
        ...     )
        ... )
        ```

        * Fetch data:

        ```pycon
        >>> data = vbt.TVData.fetch(
        ...     "NASDAQ:AAPL",
        ...     timeframe="1 hour"
        ... )
        ```
    """

    _setting_keys: tp.SettingsKeys = dict(custom="data.custom.tv")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        market: tp.Optional[str] = None,
        text: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        client: tp.Optional[TVClient] = None,
        client_config: tp.DictLike = None,
        delay: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all symbols.

        Uses market scanner when `market` is provided (returns all symbols, big payload)
        Uses symbol search when either `text` or `exchange` is provided (returns a subset of symbols)."""
        tv_cfg = cls.get_settings(key_id="custom")

        if market is None and text is None and exchange is None:
            raise ValueError("Please provide either market, or text and/or exchange")
        if market is not None and (text is not None or exchange is not None):
            raise ValueError("Please provide either market, or text and/or exchange")
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        if delay is None:
            delay = tv_cfg["delay"]
        if show_progress is None:
            show_progress = tv_cfg["show_progress"]
        pbar_kwargs = merge_dicts(tv_cfg["pbar_kwargs"], pbar_kwargs)

        if market is None:
            data = client.search_symbol(
                text=text,
                exchange=exchange,
                delay=delay,
                show_progress=show_progress,
                pbar_kwargs=pbar_kwargs,
            )
            all_symbols = map(lambda x: x["exchange"] + ":" + x["symbol"], data)
        else:
            data = client.scan_symbols(market.lower())
            all_symbols = map(lambda x: x["s"], data)
        found_symbols = []
        for symbol in all_symbols:
            if pattern is not None:
                if not cls.key_match(symbol.split(":")[1], pattern, use_regex=use_regex):
                    continue
            found_symbols.append(symbol)
        return sorted(found_symbols)

    @classmethod
    def resolve_client(cls, client: tp.Optional[TVClient] = None, **client_config) -> TVClient:
        """Resolve the client.

        If provided, must be of the type `vectorbtpro.data.tv.TVClient`.
        Otherwise, will be created using `client_config`."""
        tv_cfg = cls.get_settings(key_id="custom")

        if client is None:
            client = tv_cfg["client"]
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = merge_dicts(tv_cfg["client_config"], client_config)
        if client is None:
            client = TVClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[TVClient] = None,
        client_config: tp.KwargsLike = None,
        exchange: tp.Optional[str] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.Optional[tp.TimezoneLike] = None,
        fut_contract: tp.Optional[int] = None,
        adjustment: tp.Optional[str] = None,
        extended_session: tp.Optional[bool] = None,
        pro_data: tp.Optional[bool] = None,
        limit: tp.Optional[int] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from TradingView.

        Args:
            symbol (str): Symbol.

                Symbol must be in the `EXCHANGE:SYMBOL` format if `exchange` is None.
            client (vectorbtpro.data.tv.TVClient): Client.

                See `TVData.resolve_client`.
            client_config (dict): Client config.

                See `TVData.resolve_client`.
            exchange (str): Exchange.

                Can be omitted if already provided via `symbol`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            fut_contract (int): None for cash, 1 for continuous current contract in front,
                2 for continuous next contract in front.
            adjustment (str): Adjustment.

                Either "splits" (default) or "dividends".
            extended_session (bool): Regular session if False, extended session if True.
            pro_data (bool): Whether to use pro data.
            limit (int): The maximum number of returned items.

        For defaults, see `custom.tv` in `vectorbtpro._settings.data`.
        """
        tv_cfg = cls.get_settings(key_id="custom")

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)
        if exchange is None:
            exchange = tv_cfg["exchange"]
        if timeframe is None:
            timeframe = tv_cfg["timeframe"]
        if tz is None:
            tz = tv_cfg["tz"]
        if fut_contract is None:
            fut_contract = tv_cfg["fut_contract"]
        if adjustment is None:
            adjustment = tv_cfg["adjustment"]
        if extended_session is None:
            extended_session = tv_cfg["extended_session"]
        if pro_data is None:
            pro_data = tv_cfg["pro_data"]
        if limit is None:
            limit = tv_cfg["limit"]

        freq = prepare_freq(timeframe)
        if not isinstance(timeframe, str):
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        split = split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        multiplier, unit = split
        if unit == "s":
            interval = f"{str(multiplier)}S"
        elif unit == "t":
            interval = str(multiplier)
        elif unit == "h":
            interval = f"{str(multiplier)}H"
        elif unit == "d":
            interval = f"{str(multiplier)}D"
        elif unit == "W":
            interval = f"{str(multiplier)}W"
        elif unit == "M":
            interval = f"{str(multiplier)}M"
        else:
            raise ValueError(f"Invalid timeframe '{timeframe}'")

        df = client.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            fut_contract=fut_contract,
            adjustment=adjustment,
            extended_session=extended_session,
            pro_data=pro_data,
            limit=limit,
        )
        df.rename(
            columns={
                "symbol": "Symbol",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is None:
            df = df.tz_localize("UTC")

        if "Symbol" in df:
            del df["Symbol"]
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

        return df, dict(tz_convert=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


TVData.override_feature_config_doc(__pdoc__)
