---
title: MTF analysis
description: Learn about multiple time frame analysis. Not for sharing.
---

# :material-lock-open: MTF analysis

By limiting ourselves to only one time frame, we may lose sight of the larger trend, miss clear 
levels of support and resistance, and overlook high probability entry and stop levels. Monitoring 
the same pair under different time frames (or time compressions) can help us identify the overall 
flow of an asset ([the trend is your friend](https://www.investopedia.com/articles/forex/05/050505.asp), 
after all) and key chart patterns. In fact, all technical indicators will show different results when 
used in certain times, and all those results combined can make us draw a more complete picture of the 
market we're participating in.

## Resampling

Since vectorbt is all about time series, the main operation that allows us to switch between different 
time frames is called _resampling_. There are two types of resampling: upsampling and downsampling.

[Upsampling](https://en.wikipedia.org/wiki/Upsampling) brings a time series to a shorter time frame (i.e., 
a higher frequency), such as by converting a daily price to an hourly price. The prefix "up" here means
an increase in the number of data points. This operation isn't associated with any information loss since 
none of the data is removed, just re-indexed: the value at each day appears at the very first hour in 
the upsampled array, while all other hours contain NaN. By forward-filling those NaN values, we would 
be able to compare any daily time series with an hourly time series!

```mermaid
flowchart LR;
    subgraph daily [ ]
    id1["2020-01-01"]
    end
    subgraph hourly [ ]
    id2["2020-01-01 00:00:00"]
    id3["2020-01-01 01:00:00"]
    id4["..."]
    id5["2020-01-01 23:00:00"]
    
    id1 -->|"value"| id2;
    id1 -->|"NaN"| id3;
    id1 -->|"NaN"| id4;
    id1 -->|"NaN"| id5;
    end
```

(Reload the page if the diagram doesn't show up)

[Downsampling](https://en.wikipedia.org/wiki/Downsampling_(signal_processing)), on the other hand,
brings a time series to a longer time frame (i.e., a lower frequency), such as by converting an hourly
price to a daily price. The prefix "down" here means a decrease in the number of data points. In contrast to 
upsampling, downsampling **results in information loss** since multiple pieces of information are 
aggregated into a single one. That's why time frames are also referred to as time compressions. But even 
though we lose some information, we can now observe a bigger trend!

```mermaid
flowchart LR;
    subgraph daily [ ]
    id5["2020-01-01"]
    end
    subgraph hourly [ ]
    id1["2020-01-01 00:00:00"]
    id2["2020-01-01 01:00:00"]
    id3["..."]
    id4["2020-01-01 23:00:00"]
    
    id1 -->|"value_0"| id5;
    id2 -->|"value_1"| id5;
    id3 -->|"..."| id5;
    id4 -->|"value_n"| id5;
    end
```

!!! hint
    Downsampling is a similar concept to a moving average, which aggregates information at each time 
    step to reveal a bigger trend.

## Data

Before pulling any data, we need to ask ourselves: _"What is the shortest time frame we want to analyze?"_
Once this question is answered, we need to pull the data of this exact granularity. For example,
to be able to work with the time frames `H1` (1 hour), `H4` (4 hours), and `D1` (1 day), we need data 
with at least the time frame `H1`, which can be later downsampled to derive the `H4` and `D1` time frames.

!!! note
    This wouldn't work the other way around: we cannot upsample `H4` or `D1` to derive `H1` since
    most data points would just become NaN.

```pycon
>>> import vectorbtpro as vbt

>>> h1_data = vbt.BinanceData.fetch(
...     "BTCUSDT", 
...     start="2020-01-01 UTC", 
...     end="2021-01-01 UTC",
...     timeframe="1h"
... )
```

[=100% "Period 18/18"]{: .candystripe}

Let's persist the data locally to avoid re-fetching it every time we start a new runtime:

```pycon
>>> h1_data.to_hdf()
```

We can then access the saved data easily using [HDFData](/api/data/custom/#vectorbtpro.data.custom.HDFData):

```pycon
>>> h1_data = vbt.HDFData.fetch("BinanceData.h5")
```

Let's take a look at the index of the data:

```pycon
>>> h1_data.wrapper.index  # (1)!
DatetimeIndex(['2020-01-01 00:00:00+00:00', '2020-01-01 01:00:00+00:00',
               '2020-01-01 02:00:00+00:00', '2020-01-01 03:00:00+00:00',
               '2020-01-01 04:00:00+00:00', '2020-01-01 05:00:00+00:00',
               ...
               '2020-12-31 18:00:00+00:00', '2020-12-31 19:00:00+00:00',
               '2020-12-31 20:00:00+00:00', '2020-12-31 21:00:00+00:00',
               '2020-12-31 22:00:00+00:00', '2020-12-31 23:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', length=8767, freq=None)
```

1. [Data](/api/data/base/#vectorbtpro.data.base.Data), like many other vectorbt classes,
holds a wrapper, which contains useful metadata about index and columns

As expected, the index starts at midnight of 1st January and ends at 11 PM on December 31.
But what about `freq=None`? Pandas wasn't able to derive the frequency of data because some data 
points seem to be missing. This happens relatively often and indicates that the exchange was down.
To get all the missing indices, we need to create a resampler of type 
[Resampler](/api/base/resampling/base/#vectorbtpro.base.resampling.base.Resampler) and then
use [Resampler.index_difference](/api/base/resampling/base/#vectorbtpro.base.resampling.base.Resampler.index_difference)
with `reverse=True`:

```pycon
>>> h1_resampler = h1_data.wrapper.create_resampler("1h")  # (1)!
>>> h1_resampler.index_difference(reverse=True)  # (2)!
DatetimeIndex(['2020-02-09 02:00:00+00:00', '2020-02-19 12:00:00+00:00',
               '2020-02-19 13:00:00+00:00', '2020-02-19 14:00:00+00:00',
               '2020-02-19 15:00:00+00:00', '2020-02-19 16:00:00+00:00',
               '2020-03-04 10:00:00+00:00', '2020-04-25 02:00:00+00:00',
               '2020-04-25 03:00:00+00:00', '2020-06-28 02:00:00+00:00',
               '2020-06-28 03:00:00+00:00', '2020-06-28 04:00:00+00:00',
               '2020-11-30 06:00:00+00:00', '2020-12-21 15:00:00+00:00',
               '2020-12-21 16:00:00+00:00', '2020-12-21 17:00:00+00:00',
               '2020-12-25 02:00:00+00:00'],
              dtype='datetime64[ns, UTC]', name='Open time', freq=None)
```

1. Resampler is simply a mapper between the index in the wrapper (source index) 
and the resampled index (target index)
2. Get all indices in the target index not present in the source index

Those are the time periods when Binance was supposedly down. The good news is: we don't need
to set those data points to NaN since vectorbt accepts missing indices just fine. In fact, marking 
those points as missing would only inflate the data and make working with indicators that don't like 
missing data much, such as TA-Lib, prone to errors.

Now, how do we downsample that data to `H4` and `D1`? If we look at the columns stored in the data 
instance, we'd see very familiar column names `Open`, `High`, `Low`, `Close`, and `Volume`:

```pycon
>>> h1_data.wrapper.columns
Index(['Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote volume',
       'Number of trades', 'Taker base volume', 'Taker quote volume'],
      dtype='object')
```

First, let's remove columns that aren't much interesting to us right now:

```pycon
>>> h1_ohlcv_data = h1_data[["Open", "High", "Low", "Close", "Volume"]]
```

The most conventional way to resample any OHLCV data is by using Pandas:

```pycon
>>> h4_ohlcv = h1_ohlcv_data.get().resample("4h").agg({  # (1)!
...     "Open": "first",
...     "High": "max",
...     "Low": "min",
...     "Close": "last",
...     "Volume": "sum"
... })
>>> h4_ohlcv
                               Open      High       Low     Close  \\
Open time                                                           
2020-01-01 00:00:00+00:00   7195.24   7245.00   7175.46   7225.01   
2020-01-01 04:00:00+00:00   7225.00   7236.27   7199.11   7209.83   
2020-01-01 08:00:00+00:00   7209.83   7237.73   7180.00   7197.20   
...                             ...       ...       ...       ...   
2020-12-31 12:00:00+00:00  28910.29  28989.03  27850.00  28770.00   
2020-12-31 16:00:00+00:00  28782.01  29000.00  28311.00  28897.83   
2020-12-31 20:00:00+00:00  28897.84  29169.55  28780.00  28923.63   

                                 Volume  
Open time                                
2020-01-01 00:00:00+00:00   2833.749180  
2020-01-01 04:00:00+00:00   2061.295051  
2020-01-01 08:00:00+00:00   3166.654361  
...                                 ...  
2020-12-31 12:00:00+00:00  19597.147389  
2020-12-31 16:00:00+00:00  10279.179141  
2020-12-31 20:00:00+00:00   7875.879035  

[2196 rows x 5 columns]
```

1. See [Resampler.aggregate](https://pandas.pydata.org/docs/reference/api/pandas.core.resample.Resampler.aggregate.html)

We see that the time interval has increased from 1 hour to 4 hours; in fact, we just built 1 bigger 
bar out of 4 smaller ones:

```pycon
>>> h1_ohlcv_data.get().iloc[:4]
                              Open     High      Low    Close      Volume
Open time                                                                
2020-01-01 00:00:00+00:00  7195.24  7196.25  7175.46  7177.02  511.814901
2020-01-01 01:00:00+00:00  7176.47  7230.00  7175.71  7216.27  883.052603
2020-01-01 02:00:00+00:00  7215.52  7244.87  7211.41  7242.85  655.156809
2020-01-01 03:00:00+00:00  7242.66  7245.00  7220.00  7225.01  783.724867

>>> h4_ohlcv.iloc[[0]]
                              Open    High      Low    Close      Volume
Open time                                                               
2020-01-01 00:00:00+00:00  7195.24  7245.0  7175.46  7225.01  2833.74918
```

Great! But as with everything, vectorbt deploys special methods that either do such things more 
efficiently or more flexibly (mostly both). 

Remember how most classes in vectorbt subclass [Analyzable](/api/generic/analyzable/#vectorbtpro.generic.analyzable.Analyzable)? 
In turn, this class subclasses [Wrapping](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping), which 
is designed for managing all the Pandas objects stored in a class instance. Since it also contains the 
Pandas metadata such as index and columns, we can use that index for resampling. Particularly, any 
subclass of [Wrapping](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping) has an abstract method 
[Wrapping.resample](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping.resample),
which can be overridden to resample complex vectorbt objects, such as instances of
[Data](/api/data/base/#vectorbtpro.data.base.Data) and 
[Portfolio](/api/portfolio/base/#vectorbtpro.portfolio.base.Portfolio).

Luckily for us, vectorbt has implemented this method in most classes that can actually be resampled.
In most cases, it forwards most arguments and keyword arguments to 
[Wrapping.create_resampler](/api/base/wrapping/#vectorbtpro.base.wrapping.Wrapping.create_resampler)
to build a resampler, and then applies this resampler on all Pandas objects stored in a vectorbt object. 
Continuing with data, [Data.resample](/api/data/base/#vectorbtpro.data.base.Data.resample) looks for 
any OHLCV columns in a data instance and resamples them automatically. But what happens with other columns, 
such as `Number of trades`? Their resampling function can be defined in the column config 
[Data.column_config](/api/data/base/#vectorbtpro.data.base.Data.column_config). Even better:
vectorbt has defined resampling functions for all columns of all remote data classes!

Let's take a look at the column config of [BinanceData](/api/data/custom/#vectorbtpro.data.custom.BinanceData):

```pycon
>>> print(vbt.prettify(vbt.BinanceData.column_config))
Config({
    'Close time': {
        'resample_func': <function BinanceData.<lambda> at 0x7fd2d60c4378>
    },
    'Quote volume': {
        'resample_func': <function BinanceData.<lambda> at 0x7fd2d60c4400>
    },
    'Number of trades': {
        'resample_func': <function BinanceData.<lambda> at 0x7fd2d60c4488>
    },
    'Taker base volume': {
        'resample_func': <function BinanceData.<lambda> at 0x7fd2d60c4510>
    },
    'Taker quote volume': {
        'resample_func': <function BinanceData.<lambda> at 0x7fd2d60c4598>
    }
})
```

Each of those lambda functions takes the Pandas object and the resampler, and performs the operation using 
[GenericAccessor.resample_apply](/api/generic/accessors/#vectorbtpro.generic.accessors.GenericAccessor.resample_apply).

!!! hint
    There is no need to define resampling functions for OHLCV columns as vectorbt already knows what to do.

Let's downsample `H1` to `H4` and `D1` with a single line of code:

```pycon
>>> h1_data.use_column_config_of(vbt.BinanceData) # (1)!

>>> h4_data = h1_data.resample("4h")
>>> d1_data = h1_data.resample("1d")
```

1. [HDFData](/api/data/custom/#vectorbtpro.data.custom.HDFData) stores information from Binance
but doesn't know about it, thus let's use the column config from 
[BinanceData](/api/data/custom/#vectorbtpro.data.custom.BinanceData). Note that this is only
required for data classes that aren't tied to a specific column layout.

That's it!

```pycon
>>> d1_data.get().iloc[[0, -1]]  # (1)!
                               Open     High       Low     Close  \\
Open time                                                          
2020-01-01 00:00:00+00:00   7195.24   7255.0   7175.15   7200.85   
2020-12-31 00:00:00+00:00  28875.55  29300.0  27850.00  28923.63   

                                 Volume                          Close time  \\
Open time                                                                     
2020-01-01 00:00:00+00:00  16792.388165 2020-01-01 23:59:59.999000064+00:00   
2020-12-31 00:00:00+00:00  75508.505152 2020-12-31 23:59:59.999000064+00:00   

                           Quote volume  Number of trades  Taker base volume  \\
Open time                                                                      
2020-01-01 00:00:00+00:00  1.212145e+08            194010        8946.955535   
2020-12-31 00:00:00+00:00  2.173600e+09           1552793       36431.622080   

                           Taker quote volume  
Open time                                      
2020-01-01 00:00:00+00:00        6.459779e+07  
2020-12-31 00:00:00+00:00        1.049389e+09 
```

1. Get the first and the last row of the downsampled data

We can validate the results from resampling by comparing them against the same time frame 
fetched directly from Binance:

```pycon
>>> vbt.BinanceData.fetch(
...     "BTCUSDT", 
...     start="2020-01-01 UTC", 
...     end="2021-01-01 UTC",
...     timeframe="1d"
... ).get().iloc[[0, -1]]
```

[=100% "100%"]{: .candystripe}

```pycon
                               Open     High       Low     Close  \\
Open time                                                          
2020-01-01 00:00:00+00:00   7195.24   7255.0   7175.15   7200.85   
2020-12-31 00:00:00+00:00  28875.55  29300.0  27850.00  28923.63   

                                 Volume                       Close time  \\
Open time                                                                  
2020-01-01 00:00:00+00:00  16792.388165 2020-01-01 23:59:59.999000+00:00   
2020-12-31 00:00:00+00:00  75508.505152 2020-12-31 23:59:59.999000+00:00   

                           Quote volume  Number of trades  Taker base volume  \\
Open time                                                                      
2020-01-01 00:00:00+00:00  1.212145e+08            194010        8946.955535   
2020-12-31 00:00:00+00:00  2.173600e+09           1552793       36431.622080   

                           Taker quote volume  
Open time                                      
2020-01-01 00:00:00+00:00        6.459779e+07  
2020-12-31 00:00:00+00:00        1.049389e+09  
```

Our data instance just resampled itself the same way as done by Binance :fire:

[:material-lock: Notebook](https://github.com/polakowo/vectorbt.pro/blob/main/locked-notebooks.md){ .md-button target="blank_" }
