---
title: Remote
description: Documentation on handling remote data
icon: material/web
---

# :material-web: Remote

Data classes that subclass [RemoteData](/api/data/custom/#vectorbtpro.data.custom.RemoteData) specialize
in pulling (mostly OHLCV) data from remote data sources. In contrast to the classes for locally stored data,
they communicate with remote API endpoints and are subject to authentication, authorization, throttling, 
and other mechanisms that must be taken into account. Also, the amount of data to be fetched is
usually not known in advance, and because most data providers have API rate limits and can return only
a limited amount of data for each incoming request, there is often a need to iterate over smaller bunches 
of data and properly concatenate them. Fortunately, vectorbt implements a number of preset data classes 
that can do all the jobs above automatically.

## Arguments

Most remote data classes have the following arguments in common:

| Argument           | Description                                                                                                                                                                                                                                                                                                                                                       |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `client`           | Client object required to make a request. Usually, the data class implements an additional class method `resolve_client` to instantiate the client based on the keyword arguments in `client_config` if the client is `None`. If the client has been provided, this step is omitted. You don't need to call the method `resolve_client`, it's done automatically. |
| `client_config`    | Keyword arguments used to instantiate the client.                                                                                                                                                                                                                                                                                                                 |
| `start`            | Start datetime. Will be converted into a `datetime.datetime` using [to_tzaware_datetime](/api/utils/datetime_/#vectorbtpro.utils.datetime_.to_tzaware_datetime) and may be further post-processed to fit the format accepted by the data provider.                                                                                                                |
| `end`              | End datetime. Will be converted into a `datetime.datetime` using [to_tzaware_datetime](/api/utils/datetime_/#vectorbtpro.utils.datetime_.to_tzaware_datetime) and may be further post-processed to fit the format accepted by the data provider.                                                                                                                  |
| `timeframe`        | Timeframe supplied as a human-readable string (such as `1 day`) consisting of a multiplier (`1`) and a unit (`day`). Will be parsed into a standardized format using [split_freq_str](/api/utils/datetime_/#vectorbtpro.utils.datetime_.split_freq_str).                                                                                                          |
| `limit`            | Maximum number of data items to return per request.                                                                                                                                                                                                                                                                                                               |
| `delay`            | Delay in milliseconds between requests. Helps to deal with API rate limits.                                                                                                                                                                                                                                                                                       | 
| `retries`          | Number of retries in case of connectivity and other request-specific issues. Usually, only applied if the data class is capable of collecting data in bunches.                                                                                                                                                                                                    |
| `show_progress`    | Whether to shop the progress bar using [get_pbar](/api/utils/pbar/#vectorbtpro.utils.pbar.get_pbar). Usually, only applied if the data class is capable of collecting data in bunches.                                                                                                                                                                            | 
| `pbar_kwargs`      | Keyword arguments used for setting up the progress bar.                                                                                                                                                                                                                                                                                                           |
| `silence_warnings` | Whether to silence all warnings to avoid being overflooded with messages such as in case of timeouts.                                                                                                                                                                                                                                                             |
| `exchange`         | Exchange to fetch from in case the data class supports multiple. If the data class supports multiple exchanges, settings can also be defined per exchange!                                                                                                                                                                                                        |

To get the list of arguments accepted by the fetcher of a remote data class, we can look into the API 
reference, use the Python's `help` command, or the vectorbt's own helper function
[format_func](https://vectorbt.pro/api/utils/formatting/#vectorbtpro.utils.formatting.format_func) 
on the class method [Data.fetch_symbol](/api/data/base/#vectorbtpro.data.base.Data.fetch_symbol),
which creates a query for just one symbol and returns a Series/DataFrame:

```pycon
>>> import vectorbtpro as vbt
>>> import pandas as pd
>>> import numpy as np

>>> print(vbt.format_func(vbt.CCXTData.fetch_symbol))
CCXTData.fetch_symbol(
    symbol,
    exchange=None,
    exchange_config=None,
    start=None,
    end=None,
    timeframe=None,
    limit=None,
    delay=None,
    retries=None,
    fetch_params=None,
    show_progress=None,
    pbar_kwargs=None,
    silence_warnings=None
):
    Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from CCXT.
    
    Args:
        symbol (str): Symbol.
        exchange (str or object): Exchange identifier or an exchange object.
    
            See `CCXTData.resolve_exchange`.
        exchange_config (dict): Exchange config.
    
            See `CCXTData.resolve_exchange`.
        start (any): Start datetime.
    
            See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
        end (any): End datetime.
    
            See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
        timeframe (str): Timeframe.
    
            Allows human-readable strings such as "15 minutes".
        limit (int): The maximum number of returned items.
        delay (float): Time to sleep after each request (in milliseconds).
    
            !!! note
                Use only if `enableRateLimit` is not set.
        retries (int): The number of retries on failure to fetch data.
        fetch_params (dict): Exchange-specific keyword arguments passed to `fetch_ohlcv`.
        show_progress (bool): Whether to show the progress bar.
        pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.get_pbar`.
        silence_warnings (bool): Whether to silence all warnings.
    
    For defaults, see `custom.ccxt` in `vectorbtpro._settings.data`.
    Global settings can be provided per exchange id using the `exchanges` dictionary.
```

!!! hint

    The class method [Data.fetch](/api/data/base/#vectorbtpro.data.base.Data.fetch) usually takes the 
    same arguments as [Data.fetch_symbol](/api/data/base/#vectorbtpro.data.base.Data.fetch_symbol).

As we can see, the class [CCXTData](/api/data/custom/#vectorbtpro.data.custom.CCXTData) takes 
the exchange object, the timeframe, the start date, the end date, and other keyword arguments. 

### Settings

But why are all argument values `None`? Remember that `None` has a special meaning and instructs 
vectorbt to pull the argument's default value from the [global settings](/api/_settings/). 
Particularly, we should look into the settings defined for CCXT, which are located in the dictionary 
under `custom.ccxt` in [settings.data](/api/_settings/#vectorbtpro._settings.data):

```pycon
>>> print(vbt.prettify(vbt.settings.data['custom']['ccxt']))
FrozenConfig(
    exchange='binance',
    exchange_config=dict(
        enableRateLimit=True
    ),
    start=0,
    end='now UTC',
    timeframe='1d',
    limit=1000,
    delay=None,
    retries=3,
    show_progress=True,
    pbar_kwargs=dict(),
    fetch_params=dict(),
    exchanges=dict(),
    silence_warnings=False
)
```

Another way to get the settings is by using the method 
[Data.get_settings](/api/data/base/#vectorbtpro.data.base.Data.get_settings):

```pycon
>>> print(vbt.prettify(vbt.CCXTData.get_settings(key_id="custom")))
dict(
    exchange='binance',
    exchange_config=dict(
        enableRateLimit=True
    ),
    start=0,
    end='now UTC',
    timeframe='1d',
    limit=1000,
    delay=None,
    retries=3,
    show_progress=True,
    pbar_kwargs=dict(),
    fetch_params=dict(),
    exchanges=dict(),
    silence_warnings=False
)
```

!!! hint
    Data classes register two key ids: `base` and `custom`. The id `base` manipulates the settings
    for the base class [Data](/api/data/base/#vectorbtpro.data.base.Data), while the id `custom` 
    manipulates the settings for any subclass of the class [CustomData](/api/data/custom/#vectorbtpro.data.custom.CustomData).

Using the default arguments will pull the symbol's entire daily history from Binance.

To set any default, we can change the config directly. Let's change the exchange to BitMEX:

```pycon
>>> vbt.settings.data["custom"]["ccxt"]["exchange"] = "bitmex"
```

Even simpler: similarly to how we used the method 
[Data.get_settings](/api/data/base/#vectorbtpro.data.base.Data.get_settings)
to get the settings dictionary, let's use the method 
[Data.set_settings](/api/data/base/#vectorbtpro.data.base.Data.set_settings)
to set them:

```pycon
>>> vbt.CCXTData.set_settings(key_id="custom", exchange="bitmex")
>>> vbt.settings.data["custom"]["ccxt"]["exchange"]
'bitmex'
```

!!! note
    Overriding keys in the dictionary returned by 
    [Data.get_settings](/api/data/base/#vectorbtpro.data.base.Data.get_settings)
    will have no effect.

What if we messed up? No need to panic! We can reset the settings at any time:

```pycon
>>> vbt.CCXTData.reset_settings(key_id="custom")
>>> vbt.settings.data["custom"]["ccxt"]["exchange"]
'binance'
```

!!! hint
    This won't reset all settings in vectorbt, only those corresponding to this particular class.

### Start and end

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
...     timeframe='1m'
... )
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

### Timeframe

The timeframe format has been standardized across the entire vectorbt codebase, including the preset data 
classes. This is done by the function [split_freq_str](/api/utils/datetime_/#vectorbtpro.utils.datetime_.split_freq_str),
which splits a timeframe string into a multiplier and a unit:

```pycon
>>> from vectorbtpro.utils.datetime_ import split_freq_str

>>> split_freq_str("15 minutes")
(15, 't')

>>> split_freq_str("daily")
(1, 'd')

>>> split_freq_str("1wk")
(1, 'W')

>>> split_freq_str("annually")
(1, 'Y')
```

After the split, each preset data class transforms the resulting multiplier and the unit into a 
format acceptable by its API. For example, in the class [PolygonData](/api/data/custom/#vectorbtpro.data.custom.PolygonData),
the unit `t` is getting translated into `minute`, while in the class [CCXTData](/api/data/custom/#vectorbtpro.data.custom.CCXTData)
it's getting translated into `m`. But why is the unit `t` instead of `m` in the first place?
This has something to do with date offsets: since timeframes are not only used in data classes
but also to resample and group data, we require the unit to be accepted as both a date offset
and a timedelta, unambiguously. For example, using `m` to construct a date offset (for the use in 
[pandas.DataFrame.resample](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html))
would yield a month end, while using it to construct a timedelta would yield a minute:

```pycon
>>> from pandas.tseries.frequencies import to_offset

>>> to_offset("1m")
<MonthEnd>

>>> pd.Timedelta("1m")
Timedelta('0 days 00:01:00')
```

The unit `t`, on the other hand, is understood as a minute by both:

```pycon
>>> to_offset("1t")
<Minute>

>>> pd.Timedelta("1t")
Timedelta('0 days 00:01:00')
```

Let's pull the 30-minute `BTC/USDT` data of the current day:

```pycon
>>> ccxt_data = vbt.CCXTData.fetch(
...     'BTC/USDT',
...     start="today midnight UTC", 
...     timeframe='30 minutes'
... )
>>> ccxt_data.get()
                               Open      High       Low     Close      Volume
Open time                                                                    
2022-08-03 00:00:00+00:00  22985.93  23079.39  22784.80  22816.58  3817.29442
2022-08-03 00:30:00+00:00  22817.93  22881.22  22727.00  22793.56  3404.20062
2022-08-03 01:00:00+00:00  22793.57  22796.76  22681.22  22761.79  3097.42224
...                             ...       ...       ...       ...         ...
2022-08-03 09:00:00+00:00  23293.27  23399.99  23286.24  23326.36  4365.28629
2022-08-03 09:30:00+00:00  23326.36  23400.00  23316.86  23383.55  2637.23450
2022-08-03 10:00:00+00:00  23385.48  23453.35  23351.42  23409.76  2988.63481
```

### Client

Many APIs require a client to make a request. Data classes based on such APIs usually have a 
class method with the name `resolve_client` for resolving the client, which is called before 
pulling each symbol. If the client hasn't been provided by the user (`None`), this method creates one 
automatically based on the config `client_config`. Such a config can contain various things: from API 
keys to connection parameters. For example, let's take a look at the default client of 
[BinanceData](/api/data/custom/#vectorbtpro.data.custom.BinanceData):

```pycon
>>> binance_client = vbt.BinanceData.resolve_client()
>>> binance_client
<binance.client.Client at 0x7f893a193af0>
```

To supply information to this client, we can provide keyword arguments directly:

```pycon
>>> binance_client = vbt.BinanceData.resolve_client(
...     api_key="YOUR_KEY",
...     api_secret="YOUR_SECRET"
... )
>>> binance_client
<binance.client.Client at 0x7f89183512e0>
```

Since the client is getting created automatically, we can pass all the client-related
information using the argument `client_config` during fetching:

```pycon
>>> binance_data = vbt.BinanceData.fetch(
...     "BTCUSDT",
...     client_config=dict(
...         api_key="YOUR_KEY",
...         api_secret="YOUR_SECRET"
...     )
... )
>>> binance_data.get()
                               Open      High       Low     Close  \\
Open time                                                           
2017-08-17 00:00:00+00:00   4261.48   4485.39   4200.74   4285.08   
2017-08-18 00:00:00+00:00   4285.08   4371.52   3938.77   4108.37   
2017-08-19 00:00:00+00:00   4108.37   4184.69   3850.00   4139.98   
...                             ...       ...       ...       ...   
2022-08-01 00:00:00+00:00  23296.36  23509.68  22850.00  23268.01   
2022-08-02 00:00:00+00:00  23266.90  23459.89  22654.37  22987.79   
2022-08-03 00:00:00+00:00  22985.93  23453.35  22681.22  23429.08   
                                ...
                           Taker base volume  Taker quote volume  
Open time                                                         
2017-08-17 00:00:00+00:00         616.248541        2.678216e+06  
2017-08-18 00:00:00+00:00         972.868710        4.129123e+06  
2017-08-19 00:00:00+00:00         274.336042        1.118002e+06  
...                                      ...                 ...  
2022-08-01 00:00:00+00:00       71458.395830        1.658446e+09  
2022-08-02 00:00:00+00:00       78122.085010        1.794828e+09  
2022-08-03 00:00:00+00:00       28391.646330        6.536598e+08  

[1813 rows x 9 columns]
```

But if you run [BinanceData.resolve_client](/api/data/custom/#vectorbtpro.data.custom.BinanceData.resolve_client),
you'd know that it takes time to instantiate a client, and we don't want to wait that long for every
single symbol we're attempting to fetch. Thus, a better decision would be instantiating a
client manually only once and then passing it via the argument `client`, which will reuse the client
and make fetching noticeably faster:

```pycon
>>> binance_data = vbt.BinanceData.fetch(
...     "BTCUSDT",
...     client=binance_client
... )
```

!!! info
    This will also enable re-using the client or the client config during updating since passing any 
    argument to the fetcher will store it inside the dictionary 
    [Data.fetch_kwargs](/api/data/base/#vectorbtpro.data.base.Data.fetch_kwargs),
    which is used by the updater.

!!! warning
    But this also means that sharing the data object with anyone may expose your credentials!

To not compromise the security, the recommended approach is to set any credentials and clients
globally, as we discussed previously. This won't store them inside the data instance.

```pycon
>>> vbt.BinanceData.set_settings(
...     key_id="custom",
...     client_config=dict(
...         api_key="YOUR_KEY",
...         api_secret="YOUR_SECRET"
...     )
... )
```

!!! hint
    See the API documentation of the particular data class for examples.

### Saving

To save any remote data instance, see [this documentation](/documentation/data/local/). In short:
pickling is preferred because it also saves all the arguments that were passed to the fetcher, 
such as the selected timeframe. Those arguments are important when updating - without them,
you'd have to provide them manually every time you attempt to update the data.

```pycon
>>> binance_data = vbt.BinanceData.fetch(
...     "BTCUSDT",
...     start="today midnight UTC",
...     timeframe="1 hour"
... )
>>> binance_data.save("binance_data")

>>> binance_data = vbt.BinanceData.load("binance_data")
>>> print(vbt.prettify(binance_data.fetch_kwargs))
symbol_dict(
    BTCUSDT=dict(
        start='today midnight UTC',
        timeframe='1 hour',
        show_progress=True,
        pbar_kwargs=dict(),
        silence_warnings=False
    )
)
```

As we can see, all the arguments were saved along with the data instance. But in a case
where we don't plan on updating the data, we can save the arrays themselves across one or
multiple CSV files/HDF keys, one per symbol:

```pycon
>>> binance_data.to_csv()

>>> csv_data = vbt.CSVData.fetch("BTCUSDT.csv")
>>> print(vbt.prettify(csv_data.fetch_kwargs))
symbol_dict(
    BTCUSDT=dict(
        path=PosixPath('BTCUSDT.csv')
    )
)
```

The fetching-related keyword arguments do not include the timeframe and other parameters anymore,
they include only those that are important for the current data class holding the data - 
[CSVData](/api/data/custom/#vectorbtpro.data.custom.CSVData).
But we can still switch the class of our data instance back to the original data class 
[BinanceData](/api/data/custom/#vectorbtpro.data.custom.BinanceData) using the method
[Data.switch_class](/api/data/base/#vectorbtpro.data.base.Data.switch_class). We will also
clear the fetching-related and returned keyword arguments since they are class-specific:

```pycon
>>> binance_data = csv_data.switch_class(
...     new_cls=vbt.BinanceData, 
...     clear_fetch_kwargs=True,
...     clear_returned_kwargs=True
... )
>>> type(binance_data)
vectorbtpro.data.custom.BinanceData
```

Finally, let's use [Data.update_fetch_kwargs](/api/data/base/#vectorbtpro.data.base.Data.update_fetch_kwargs)
to update the fetching-related keyword arguments with the timeframe to avoid repeatedly 
setting it when updating:

```pycon
>>> binance_data = binance_data.update_fetch_kwargs(timeframe="1 hour")
>>> print(vbt.prettify(binance_data.fetch_kwargs))
symbol_dict(
    BTCUSDT=dict(
        timeframe='1 hour'
    )
)
```

All of this could have been avoided if we used pickling.

### Updating

Updating a data instance is generally easy:

```pycon
>>> binance_data = binance_data.update()
```

!!! note
    Updating the current data instance always returns a new data instance.

Under the hood, the updater first overrides the start date with the latest date in the index,
and then calls the fetcher. That's why we can specify or override any argument originally
used in fetching. Also note that it will only pull new data if the end date is not fixed:
if we used the end date `2022-01-01` when fetching, it will be used again when updating,
thus make sure to set `end` to `"now"` or `"now UTC"`. Let's first fetch the history
for the year 2020, and then append the history for the year 2021:

```pycon
>>> binance_data = vbt.BinanceData.fetch(
...     "BTCUSDT", 
...     start="2020-01-01", 
...     end="2021-01-01"
... )
>>> binance_data = binance_data.update(end="2022-01-01")  # (1)!
>>> binance_data.wrapper.index
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-02 00:00:00+00:00',
               '2020-01-03 00:00:00+00:00', '2020-01-04 00:00:00+00:00',
               '2020-01-05 00:00:00+00:00', '2020-01-06 00:00:00+00:00',
               ...
               '2021-12-26 00:00:00+00:00', '2021-12-27 00:00:00+00:00',
               '2021-12-28 00:00:00+00:00', '2021-12-29 00:00:00+00:00',
               '2021-12-30 00:00:00+00:00', '2021-12-31 00:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', length=731, freq='D')
```

1. Without overriding, the argument `end` will default to the value passed to the fetcher - `2021-01-01`

## From URL

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

### AWS S3

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
to take advantage of the vectorbt's powerful [Data](/api/data/base/#vectorbtpro.data.base.Data) class, 
for example, to update the remote datasets whenever new data points arrive - a true :gem: