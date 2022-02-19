---
title: Scheduling
icon: material/timer-outline
---

# Scheduling

Most data sources aren't sitting idle: they steadily generate new data. To keep up with new information,
we can utilize a schedule manager (or even the simplest while-loop) to periodically run jobs related to 
data capturing and manipulation.

## Updating

We can schedule data updates easily using [DataUpdater](/api/data/updater/#vectorbtpro.data.updater.DataUpdater),
which takes a data instance of type [Data](/api/data/base/#vectorbtpro.data.base.Data) and a schedule manager 
of type [ScheduleManager](/api/utils/schedule_/#vectorbtpro.utils.schedule_.ScheduleManager), and periodically 
triggers an update that replaces the old data instance with the new one. We can then access the new instance
under [DataUpdater.data](/api/data/updater/#vectorbtpro.data.updater.DataUpdater.data). The update happens
in the method [DataUpdater.update](/api/data/updater/#vectorbtpro.data.updater.DataUpdater.update), which
can be overridden and used to run some custom logic when new data arrives. Since the updater class 
subclasses [Configured](/api/utils/config/#vectorbtpro.utils.config.Configured), it also takes care 
of replacing its config once `data` changes.

!!! important
    It's one of the few classes in vectorbt that aren't read-only. Do not rely on caching inside it!

Let's use this simple but powerful class to update and plot the last 10 minutes of a Binance 
ticker, every 10 seconds, for 5 minutes. First, we will pull the latest 10 minutes of data:

```pycon
>>> import vectorbtpro as vbt

>>> data = vbt.BinanceData.fetch(
...     'BTCUSDT', 
...     start='10 minutes ago UTC', 
...     end='now UTC', 
...     interval='1m'
... )

>>> data.get('Close')
Open time
2022-02-19 20:09:00+00:00    40005.78
2022-02-19 20:10:00+00:00    40001.80
2022-02-19 20:11:00+00:00    40006.45
2022-02-19 20:12:00+00:00    40003.68
2022-02-19 20:13:00+00:00    40022.24
2022-02-19 20:14:00+00:00    40026.73
2022-02-19 20:15:00+00:00    40048.88
2022-02-19 20:16:00+00:00    40044.92
2022-02-19 20:17:00+00:00    40044.03
2022-02-19 20:18:00+00:00    40049.93
Freq: T, Name: Close, dtype: float64
```

Then, we'll subclass [DataUpdater](/api/data/updater/#vectorbtpro.data.updater.DataUpdater)
to accept the figure and update it together with the data. Moreover, to not miss anything visually, 
after each update, we will append the figure's PNG image to a GIF file:

```pycon
>>> import imageio

>>> class OHLCFigUpdater(vbt.DataUpdater):
...     def __init__(self, data, fig, writer=None, display_last=None, 
...                  stop_after=None, **kwargs):
...         vbt.DataUpdater.__init__(  # (1)!
...             self, 
...             data, 
...             writer=writer,  # (2)!
...             display_last=display_last,
...             stop_after=stop_after,
...             **kwargs
...         )
...
...         self._fig = fig
...         self._writer = writer
...         self._display_last = display_last
...         self._stop_after = stop_after
...         self._start_dt = vbt.to_tzaware_datetime('now UTC')  # (3)!
...
...     @property  # (4)!
...     def fig(self):
...         return self._fig
...
...     @property
...     def writer(self):
...         return self._writer
...
...     @property
...     def display_last(self):
...         return self._display_last
...
...     @property
...     def stop_after(self):
...         return self._stop_after
...
...     @property
...     def start_dt(self):
...         return self._start_dt
...         
...     def update(self, **kwargs):
...         vbt.DataUpdater.update(self, **kwargs)  # (5)!
...         
...         df = self.data.get()
...         if self.display_last is not None:
...             df = df[df.index[-1] - self.display_last:]  # (6)!
...
...         trace = self.fig.data[0]
...         with self.fig.batch_update():
...             trace.x = df['Close'].index  # (7)!
...             trace.open = df['Open'].values
...             trace.high = df['High'].values
...             trace.low = df['Low'].values
...             trace.close = df['Close'].values
...
...         if self.writer is not None:
...             fig_data = imageio.imread(self.fig.to_image(format="png"))
...             self.writer.append_data(fig_data)  # (8)!
...
...         if self.stop_after is not None:
...             now_dt = vbt.to_tzaware_datetime('now UTC')
...             if now_dt - self.start_dt >= self.stop_after:
...                 raise vbt.CancelledError  # (9)!
```

1. Call the constructor of `DataUpdater`
2. Pass all class-specific keyword arguments to include them in the 
[config](/api/utils/config/#vectorbtpro.utils.config.Configured.config)
3. Register the start time
4. Properties prevent the user (and the program) from overwriting the object, which is some 
kind of convention in vectorbt
5. Call `DataUpdater.update`, otherwise, the data won't update!
6. Get the OHLC data within a specific time period (optional)
7. Update the data of the trace (see [Candlestick Charts](https://plotly.com/python/candlestick-charts/))
8. Append the data of the figure to the GIF file (optional)
9. Stop once the job has run for a specific amount of time by throwing 
[CancelledError](/api/utils/schedule_/#vectorbtpro.utils.schedule_.CancelledError) (optional)

Don't forget to enable logging (if desirable):

```pycon
>>> import logging

>>> logging.basicConfig(level = logging.INFO)
```

Finally, run the job of `OHLCFigUpdater` every 10 seconds:

```pycon
>>> import pandas as pd

>>> fig = data.plot(plot_type='candlestick', plot_volume=False)
>>> fig  # (1)!

>>> with imageio.get_writer('ohlc_fig_updater.gif', fps=3) as writer:
...     ohlc_fig_updater = OHLCFigUpdater(
...         data=data, 
...         fig=fig, 
...         writer=writer,
...         display_last=pd.Timedelta(minutes=10),
...         stop_after=pd.Timedelta(minutes=5)
...     )
...     ohlc_fig_updater.update_every(10)  # (2)!
INFO:vectorbtpro.utils.schedule_:Starting schedule manager with jobs [Every 10 seconds do update() (last run: [never], next run: 2022-02-19 21:18:38)]
INFO:vectorbtpro.data.updater:Updated data has 10 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:18:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 10 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:18:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 11 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:19:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 11 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:19:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 11 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:19:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 11 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:19:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 11 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:19:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 12 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:20:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 12 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:20:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 12 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:20:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 12 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:20:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 12 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:20:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 13 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:21:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 13 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:21:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 13 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:21:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 13 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:21:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 13 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:21:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 14 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:22:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 14 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:22:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 14 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:22:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 14 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:22:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 14 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:22:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 15 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:23:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 15 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:23:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 15 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:23:00+00:00
INFO:vectorbtpro.data.updater:Updated data has 15 rows from 2022-02-19 20:09:00+00:00 to 2022-02-19 20:23:00+00:00
INFO:vectorbtpro.utils.schedule_:Stopping schedule manager
```

1. Run these two lines in a separate cell to see the updates in real time
2. Using [DataUpdater.update_every](/api/data/updater/#vectorbtpro.data.updater.DataUpdater.update_every)

To stop earlier, simply interrupt the execution.

!!! hint
    To run the job in the background, set `in_background` to True. The execution can then be
    manually stopped by calling `ohlc_fig_updater.schedule_manager.stop()`.

After the data updater has finished running, we can access the entire data:

```pycon
>>> ohlc_fig_updater.data.get('Close')
Open time
2022-02-19 20:09:00+00:00    40005.78
2022-02-19 20:10:00+00:00    40001.80
2022-02-19 20:11:00+00:00    40006.45
2022-02-19 20:12:00+00:00    40003.68
2022-02-19 20:13:00+00:00    40022.24
2022-02-19 20:14:00+00:00    40026.73
2022-02-19 20:15:00+00:00    40048.88
2022-02-19 20:16:00+00:00    40044.92
2022-02-19 20:17:00+00:00    40044.03
2022-02-19 20:18:00+00:00    40045.36
2022-02-19 20:19:00+00:00    40047.68
2022-02-19 20:20:00+00:00    40036.74
2022-02-19 20:21:00+00:00    40037.69
2022-02-19 20:22:00+00:00    40039.92
2022-02-19 20:23:00+00:00    40041.62
Freq: T, Name: Close, dtype: float64
```

And here's the produced GIF:

![](/assets/images/ohlc_fig_updater.gif)

!!! hint
    The smallest time unit of [ScheduleManager](/api/utils/schedule_/#vectorbtpro.utils.schedule_.ScheduleManager)
    is a second. For high-precision job scheduling, use a loop with a timer.