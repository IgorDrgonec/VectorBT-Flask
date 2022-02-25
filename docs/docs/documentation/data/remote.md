---
title: Remote
icon: material/web
---

# Remote

Data classes that subclass [RemoteData](/api/data/custom/#vectorbtpro.data.custom.RemoteData) specialize
in pulling (mostly OHLCV) data from remote data sources. In contrast to the classes for locally stored data,
they communicate with remote API endpoints and are subject to authentication, authorization, throttling, 
and other mechanisms that must be taken into account. Also, the amount of data to be fetched is
usually not known in advance, and because most data providers have API rate limits and can return only
a limited amount of data for each incoming request, there is often a need to manually iterate over
smaller bunches of data and properly concatenate them.

## Preset classes

Fortunately, vectorbt implements a number of preset data classes that can do all the jobs above automatically.
To get the list of arguments accepted by the fetcher of a remote data class, we can look into the API 
reference, use the Python's `help` command, or the vectorbt's own helper function
[format_func](https://vectorbt.pro/api/utils/formatting/#vectorbtpro.utils.formatting.format_func) 
on the class method [Data.fetch_symbol](/api/data/base/#vectorbtpro.data.base.Data.fetch_symbol):

```pycon
>>> import vectorbtpro as vbt

>>> print(vbt.format_func(vbt.CCXTData.fetch_symbol))
CCXTData.fetch_symbol(
    symbol,
    exchange=None,
    timeframe=None,
    start=None,
    end=None,
    delay=None,
    limit=None,
    retries=None,
    exchange_config=None,
    fetch_params=None,
    show_progress=None,
    pbar_kwargs=None
):
    Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from CCXT.
    
    Args:
        symbol (str): Symbol.
        exchange (str or object): Exchange identifier or an exchange object of type
            `ccxt.base.exchange.Exchange`.
        timeframe (str): Timeframe supported by the exchange.
        start (any): Start datetime.
    
            See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
        end (any): End datetime.
    
            See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
        delay (float): Time to sleep after each request (in milliseconds).
    
            !!! note
                Use only if `enableRateLimit` is not set.
        limit (int): The maximum number of returned items.
        retries (int): The number of retries on failure to fetch data.
        exchange_config (dict): Keyword arguments passed to the exchange upon instantiation.
    
            Will raise an exception if exchange has been already instantiated.
        fetch_params (dict): Exchange-specific keyword arguments passed to `fetch_ohlcv`.
        show_progress (bool): Whether to show the progress bar.
        pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.
    
    For defaults, see `custom.ccxt` in `vectorbtpro._settings.data`.
    Global settings can be provided per exchange id using the `exchanges` dictionary.
```

!!! hint

    The class method [Data.fetch](/api/data/base/#vectorbtpro.data.base.Data.fetch) usually takes the 
    same arguments as [Data.fetch_symbol](/api/data/base/#vectorbtpro.data.base.Data.fetch_symbol),
    but some classes override this method to initialize their clients or for other pre-processing tasks,
    so it's worth looking at its signature too.

As we can see, [CCXTData](/api/data/custom/#vectorbtpro.data.custom.CCXTData) takes the exchange, the 
timeframe, the start date, the end date, and other keyword arguments. But why are all argument values `None`? 
Remember that `None` has a special meaning and instructs vectorbt to pull the argument's default value from the 
[global settings](/api/_settings/). Particularly, we should look into the settings defined for CCXT, 
which are located in the dictionary under `custom.ccxt` in [settings.data](/api/_settings/#vectorbtpro._settings.data):

```pycon
>>> print(vbt.prettify(vbt.settings.data['custom']['ccxt']))
{
    'exchange': 'binance',
    'exchange_config': {
        'enableRateLimit': True
    },
    'timeframe': '1d',
    'start': 0,
    'end': 'now UTC',
    'delay': None,
    'limit': 500,
    'retries': 3,
    'show_progress': True,
    'pbar_kwargs': {},
    'fetch_params': {},
    'exchanges': {}
}
```

Using the default arguments will pull the symbol's entire daily data from Binance.

Specifying dates and times is usually very easy thanks to the built-in datetime parser 
[to_tzaware_datetime](/api/utils/datetime_/#vectorbtpro.utils.datetime_.to_tzaware_datetime), which can 
parse dates and times from various objects, including human-readable strings, such as `1 day ago`:

```pycon
>>> from vectorbtpro.utils.datetime_ import to_tzaware_datetime

>>> to_tzaware_datetime('1 day ago')
datetime.datetime(2022, 2, 17, 19, 15, 19, 712250, tzinfo=datetime.timezone(datetime.timedelta(seconds=3600)))
```

Let's illustrate this by fetching the last 10 minutes of the symbols `BTC/USDT` and `ETH/USDT`:

```pycon
>>> ccxt_data = vbt.CCXTData.fetch(
...     ['BTC/USDT', 'ETH/USDT'],
...     start='10 minutes ago UTC', 
...     end='now UTC', 
...     timeframe='1m')
```

!!! note
    Different remote data classes may have different symbol notations, such as `BTC/USDT` in CCXT,
    `BTC-USD` in Yahoo Finance, `BTCUSDT` in Binance, `X:BTCUSD` in Polygon.io, etc.

[=100% "Symbol 2/2"]{: .candystripe}

```pycon
>>> ccxt_data.get('Close')
symbol                     BTC/USDT  ETH/USDT
Open time                                    
2022-02-18 18:55:00+00:00  40170.54   2795.28
2022-02-18 18:56:00+00:00  40124.78   2793.50
2022-02-18 18:57:00+00:00  40134.46   2793.07
2022-02-18 18:58:00+00:00  40173.24   2797.41
2022-02-18 18:59:00+00:00  40140.15   2794.02
2022-02-18 19:00:00+00:00  40144.75   2795.49
2022-02-18 19:01:00+00:00  40200.24   2798.01
2022-02-18 19:02:00+00:00  40209.53   2799.99
2022-02-18 19:03:00+00:00  40181.96   2798.42
2022-02-18 19:04:00+00:00  40183.94   2798.61
```

!!! hint
    Dates and times are resolved in [Data.fetch_symbol](/api/data/base/#vectorbtpro.data.base.Data.fetch_symbol).
    Whenever fetching high-frequency data, make sure to provide an already resolved start and end time
    using [to_tzaware_datetime](/api/utils/datetime_/#vectorbtpro.utils.datetime_.to_tzaware_datetime),
    otherwise, by the time the first symbol has been fetched, the resolved times for the next symbol
    may have already been changed.

Let's make this configuration the default one:

```pycon
>>> vbt.settings.data['custom']['ccxt']['start'] = '10 minutes ago UTC'
>>> vbt.settings.data['custom']['ccxt']['end'] = 'now UTC'
>>> vbt.settings.data['custom']['ccxt']['timeframe'] = '1m'

>>> ccxt_data = vbt.CCXTData.fetch(['BTC/USDT', 'ETH/USDT'])
```

[=100% "Symbol 2/2"]{: .candystripe}

To define any argument per exchange:

```pycon
>>> vbt.settings.data['custom']['ccxt']['exchanges']['binance'] = dict(retries=4)
```

This will override the universal default of `3`.

That's about CCXT, but what about other data providers? Their API is not much different.
Take the data class for communicating with [Alpaca](https://alpaca.markets/) for example:
it takes the same arguments `start`, `end`, `timeframe`, `limit`, and `exchange` but with a slightly
different behavior. Additionally, [AlpacaData](/api/data/custom/#vectorbtpro.data.custom.AlpacaData) 
overrides [Data.fetch](/api/data/base/#vectorbtpro.data.base.Data.fetch) to instantiate the client
(if not already supplied) based on keyword arguments in `client_kwargs`, such as `key_id` and `secret_key`.
Why not doing this inside [Data.fetch_symbol](/api/data/base/#vectorbtpro.data.base.Data.fetch_symbol)? 
Very simple: we don't want to repeat this for every single symbol.

```pycon
>>> alpaca_data = vbt.AlpacaData.fetch(
...     ['BTCUSD', 'ETHUSD'],
...     start='10 minutes ago UTC', 
...     end='now UTC', 
...     timeframe='1m',
...     client_kwargs=dict(
...         key_id="{API Key ID}",  # (1)!
...         secret_key="{Secret Key}",
...     )    
... )
```

1. Both parameters can also be defined globally in [settings.data](/api/_settings/#vectorbtpro._settings.data), 
or you can instantiate your own client and either pass it as `client` or define it globally as well

!!! warning
    After fetching, the secrets are kept in [Data.fetch_kwargs](/api/data/base/#vectorbtpro.data.base.Data.fetch_kwargs)
    for use in updating. Be careful when pickling the data instance and making the file accessible to others! 
    A far better approach is to define them in [settings.data](/api/_settings/#vectorbtpro._settings.data).

[=100% "Symbol 2/2"]{: .candystripe}

```pycon
>>> alpaca_data.data['BTCUSD']
                               Open      High       Low     Close     Volume  \\
timestamp                                                                      
2022-02-18 20:51:00+00:00  40076.93  40076.93  40004.96  40025.08  13.476127   
2022-02-18 20:52:00+00:00  40023.43  40023.46  39974.53  39992.62  20.641257   
2022-02-18 20:53:00+00:00  39989.30  40014.10  39975.01  39980.05  23.642147   
2022-02-18 20:54:00+00:00  39980.05  40054.64  39975.00  40004.14  25.785768   
2022-02-18 20:55:00+00:00  40004.14  40019.96  39983.61  39991.00  14.626027   
2022-02-18 20:56:00+00:00  39991.00  39991.00  39943.00  39982.76  20.511136   
2022-02-18 20:57:00+00:00  39981.37  39989.68  39925.67  39960.16  45.210278   
2022-02-18 20:58:00+00:00  39960.17  39991.62  39955.94  39966.99  21.146296   
2022-02-18 20:59:00+00:00  39967.00  40031.30  39957.65  40031.30  34.908979   

                           Trade count          VWAP  
timestamp                                             
2022-02-18 20:51:00+00:00          433  40032.038743  
2022-02-18 20:52:00+00:00          487  39992.018190  
2022-02-18 20:53:00+00:00          470  39988.574341  
2022-02-18 20:54:00+00:00          537  40013.240065  
2022-02-18 20:55:00+00:00          479  40003.483027  
2022-02-18 20:56:00+00:00          485  39965.798564  
2022-02-18 20:57:00+00:00          602  39957.348959  
2022-02-18 20:58:00+00:00          568  39973.988031  
2022-02-18 20:59:00+00:00          709  39979.397976 
```

### Common arguments

Most remote data classes have the following arguments in common:

| Argument           | Description                                                                                                                                                                                                                                                                                                                                                                                             |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `client`           | Client required to make a request. Usually, the data class overrides the method [Data.fetch](/api/data/base/#vectorbtpro.data.base.Data.fetch) to instantiate the client only once based on keyword arguments in `client_kwargs`. If `client` has been provided, this step is omitted and the client is forwarded down to [Data.fetch_symbol](/api/data/base/#vectorbtpro.data.base.Data.fetch_symbol). |
| `start`            | Start datetime. Will be converted into a `datetime.datetime` using [to_tzaware_datetime](/api/utils/datetime_/#vectorbtpro.utils.datetime_.to_tzaware_datetime) and may be further post-processed to fit the format accepted by the data provider.                                                                                                                                                      |
| `end`              | End datetime. Will be converted into a `datetime.datetime` using [to_tzaware_datetime](/api/utils/datetime_/#vectorbtpro.utils.datetime_.to_tzaware_datetime) and may be further post-processed to fit the format accepted by the data provider.                                                                                                                                                        |
| `timeframe`        | Timeframe supplied as a string (such as `1d`) consisting of a multiplier (`1`) and a timespan (`d`).                                                                                                                                                                                                                                                                                                    |
| `limit`            | Maximum number of data items to return per request.                                                                                                                                                                                                                                                                                                                                                     |
| `delay`            | Delay in milliseconds between requests. Helps to deal with API rate limits.                                                                                                                                                                                                                                                                                                                             | 
| `retries`          | Number of retries in case of connectivity and other request-specific issues. Usually, only applied if the data class is capable of collecting data in bunches.                                                                                                                                                                                                                                          |
| `show_progress`    | Whether to shop the progress bar using [get_pbar](/api/utils/pbar/#vectorbtpro.utils.pbar.get_pbar). Usually, only applied if the data class is capable of collecting data in bunches.                                                                                                                                                                                                                  | 
| `pbar_kwargs`      | Keyword arguments used for setting up the progress bar.                                                                                                                                                                                                                                                                                                                                                 |
| `silence_warnings` | Whether to silence all warnings to avoid being overflooded with messages such as in case of timeouts.                                                                                                                                                                                                                                                                                                   |
| `exchange`         | Exchange to fetch from in case the data class supports multiple.                                                                                                                                                                                                                                                                                                                                        |

All of these arguments can be set globally in [settings.data](/api/_settings/#vectorbtpro._settings.data).
When the data class supports multiple exchanges, they can also be set per exchange!

## CSV

Even though [CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData) was designed for the local
file system, we can apply a couple of tricks to pull remote data with it as well! Remember how
it uses [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)? This
function has an argument `filepath_or_buffer`, which can be a URL. All we have to do is to disable
the path matching mechanism by setting `match_paths` to False.

Here's an example of pulling S&P 500 index data:

```pycon
>>> url = "https://datahub.io/core/s-and-p-500/r/data.csv"
>>> csv_data = vbt.CSVData.fetch(url, match_paths=False)
>>> csv_data.get()
                             SP500  Dividend  Earnings  Consumer Price Index  \\
Date                                                                           
1871-01-01 00:00:00+00:00     4.44      0.26      0.40                 12.46   
1871-02-01 00:00:00+00:00     4.50      0.26      0.40                 12.84   
1871-03-01 00:00:00+00:00     4.61      0.26      0.40                 13.03   
...                            ...       ...       ...                   ...   
2018-02-01 00:00:00+00:00  2705.16     49.64       NaN                248.99   
2018-03-01 00:00:00+00:00  2702.77     50.00       NaN                249.55   
2018-04-01 00:00:00+00:00  2642.19       NaN       NaN                249.84   

                           Long Interest Rate  Real Price  Real Dividend  \\
Date                                                                       
1871-01-01 00:00:00+00:00                5.32       89.00           5.21   
1871-02-01 00:00:00+00:00                5.32       87.53           5.06   
1871-03-01 00:00:00+00:00                5.33       88.36           4.98   
...                                       ...         ...            ...   
2018-02-01 00:00:00+00:00                2.86     2714.34          49.81   
2018-03-01 00:00:00+00:00                2.84     2705.82          50.06   
2018-04-01 00:00:00+00:00                2.80     2642.19            NaN   

                           Real Earnings   PE10  
Date                                             
1871-01-01 00:00:00+00:00           8.02    NaN  
1871-02-01 00:00:00+00:00           7.78    NaN  
1871-03-01 00:00:00+00:00           7.67    NaN  
...                                  ...    ...  
2018-02-01 00:00:00+00:00            NaN  32.12  
2018-03-01 00:00:00+00:00            NaN  31.99  
2018-04-01 00:00:00+00:00            NaN  31.19  

[1768 rows x 9 columns]
```

Here's another example for AWS S3:

```pycon
>>> import boto3
>>> s3_client = boto3.client('s3')  # (1)!

>>> symbols = ['BTCUSDT', 'ETHUSDT']
>>> paths = vbt.symbol_dict({ 
...     s: s3_client.get_object(
...         Bucket='binance', 
...         Key=f'data/{s}.csv')['Body']  # (2)!
...     for s in symbols
... })
>>> s3_data = vbt.CSVData.fetch(symbols, paths=paths, match_paths=False)  # (3)!
>>> s3_data.get('Close')
symbol                      BTCUSDT  ETHUSDT
Open time                                   
2017-08-17 00:00:00+00:00   4285.08   302.00
2017-08-18 00:00:00+00:00   4108.37   293.96
2017-08-19 00:00:00+00:00   4139.98   290.91
2017-08-20 00:00:00+00:00   4086.29   299.10
2017-08-21 00:00:00+00:00   4016.00   323.29
...                             ...      ...
2022-02-14 00:00:00+00:00  42535.94  2929.75
2022-02-15 00:00:00+00:00  44544.86  3183.52
2022-02-16 00:00:00+00:00  43873.56  3122.30
2022-02-17 00:00:00+00:00  40515.70  2891.87
2022-02-18 00:00:00+00:00  39892.83  2768.74

[1647 rows x 2 columns]
```

1. See [Python, Boto3, and AWS S3: Demystified](https://realpython.com/python-boto3-aws-s3/)
2. Adapt the code to your own bucket and keys
3. Each path in `paths` will become `filepath_or_buffer`

We could have loaded both datasets using [pandas.read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
itself, but wrapping them with [CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData) allows us
to take advantage of the vectorbt's powerful data classes, for example, to update the remote datasets 
whenever new data points arrive - a true :gem: