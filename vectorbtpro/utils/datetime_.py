# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for working with dates and time."""

from datetime import datetime, timezone, timedelta, tzinfo, time

import dateparser
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytz
import re

from vectorbtpro import _typing as tp


PandasDatetimeIndex = (pd.DatetimeIndex, pd.PeriodIndex)


def freq_to_timedelta(freq: tp.FrequencyLike) -> pd.Timedelta:
    """Convert a frequency-like object to `pd.Timedelta`."""
    if isinstance(freq, pd.Timedelta):
        return freq
    if isinstance(freq, str):
        freq = "".join(freq.strip().split())
        match = re.match(r"^(\d*)([m]?)$", freq)
        if match:
            freq = match.group(1) + match.group(2)
        if re.match(r"^\d*[MyY]?$", freq):
            raise ValueError("Units 'M', 'Y' and 'y' do not represent unambiguous timedelta values")
    if isinstance(freq, str) and not freq[0].isdigit():
        # Otherwise "ValueError: unit abbreviation w/o a number"
        return pd.Timedelta(1, unit=freq)
    return pd.Timedelta(freq)


def freq_to_timedelta64(freq: tp.FrequencyLike) -> np.timedelta64:
    """Convert a frequency-like object to `np.timedelta64`."""
    if isinstance(freq, np.timedelta64):
        return freq
    if not isinstance(freq, (pd.DateOffset, pd.Timedelta)):
        freq = freq_to_timedelta(freq)
    if isinstance(freq, pd.DateOffset):
        freq = pd.Timedelta(freq)
    return freq.to_timedelta64()


def try_to_datetime_index(index: tp.IndexLike, **kwargs) -> tp.Index:
    """Try converting an index to a datetime index."""
    if not isinstance(index, pd.Index):
        if isinstance(index, str):
            try:
                index = pd.to_datetime(index)
                index = [index]
            except Exception as e:
                pass
        try:
            index = pd.Index(index)
        except Exception as e:
            index = pd.Index([index])
    if isinstance(index, pd.DatetimeIndex):
        return index
    if index.dtype == object:
        try:
            return pd.to_datetime(index, **kwargs)
        except Exception as e:
            pass
    return index


def infer_index_freq(
    index: pd.Index,
    freq: tp.Optional[tp.FrequencyLike] = None,
    allow_date_offset: bool = True,
    allow_numeric: bool = True,
    detect_via_diff: bool = False,
) -> tp.Union[None, float, tp.PandasFrequency]:
    """Infer frequency of a datetime index if `freq` is None, otherwise convert `freq`."""
    if freq is None and isinstance(index, pd.DatetimeIndex):
        if index.freqstr is not None:
            freq = to_offset(index.freqstr)
        elif index.freq is not None:
            freq = index.freq
        elif len(index) >= 3:
            freq = pd.infer_freq(index)
            if freq is not None:
                freq = to_offset(freq)
    if freq is None:
        if detect_via_diff:
            return (index[1:] - index[:-1]).min()
        return None
    if isinstance(freq, pd.Timedelta):
        return freq
    if isinstance(freq, pd.DateOffset) and allow_date_offset:
        return freq
    if isinstance(freq, (int, float)) and allow_numeric:
        return freq
    return freq_to_timedelta(freq)


def get_utc_tz() -> timezone:
    """Get UTC timezone."""
    return timezone.utc


def get_local_tz() -> timezone:
    """Get local timezone."""
    return timezone(datetime.now(timezone.utc).astimezone().utcoffset())


def convert_tzaware_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as non-naive time.

    `datetime.time` must have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).timetz()


def tzaware_to_naive_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as naive time.

    `datetime.time` must have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def naive_to_tzaware_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as non-naive time.

    `datetime.time` must not have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time().replace(tzinfo=tz_out)


def convert_naive_time(t: time, tz_out: tp.Optional[tzinfo]) -> time:
    """Return as naive time.

    `datetime.time` must not have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def is_tz_aware(dt: tp.SupportsTZInfo) -> bool:
    """Whether datetime is timezone-aware."""
    tz = dt.tzinfo
    if tz is None:
        return False
    return tz.utcoffset(datetime.now()) is not None


def to_timezone(tz: tp.TimezoneLike, to_py_timezone: tp.Optional[bool] = None, **kwargs) -> tzinfo:
    """Parse the timezone.

    Strings are parsed by `pytz` and `dateparser`, while integers and floats are treated as hour offsets.

    If `to_py_timezone` is set to True, will convert to `datetime.timezone`. See global settings.

    `**kwargs` are passed to `dateparser.parse`."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if tz is None:
        return get_local_tz()
    if to_py_timezone is None:
        to_py_timezone = datetime_cfg["to_py_timezone"]

    if isinstance(tz, str):
        try:
            tz = pytz.timezone(tz)
        except pytz.UnknownTimeZoneError:
            dt = dateparser.parse("now %s" % tz, **kwargs)
            if dt is not None:
                tz = dt.tzinfo
    if isinstance(tz, (int, float)):
        tz = timezone(timedelta(hours=tz))
    if isinstance(tz, timedelta):
        tz = timezone(tz)
    if isinstance(tz, tzinfo):
        if to_py_timezone:
            return timezone(tz.utcoffset(datetime.now()))
        return tz
    raise TypeError("Couldn't parse the timezone")


def to_tzaware_datetime(
    dt_like: tp.DatetimeLike,
    naive_tz: tp.Optional[tp.TimezoneLike] = None,
    tz: tp.Optional[tp.TimezoneLike] = None,
    **kwargs
) -> datetime:
    """Parse the datetime as a timezone-aware `datetime.datetime`.

    See [dateparser docs](http://dateparser.readthedocs.io/en/latest/) for valid string formats and `**kwargs`.

    Raw timestamps are localized to UTC, while naive datetime is localized to `naive_tz`.
    Set `naive_tz` to None to use the default value defined under `vectorbtpro._settings.datetime`.
    To explicitly convert the datetime to a timezone, use `tz` (uses `to_timezone`)."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if naive_tz is None:
        naive_tz = datetime_cfg["naive_tz"]
    if isinstance(dt_like, float):
        dt = datetime.fromtimestamp(dt_like, timezone.utc)
    elif isinstance(dt_like, int):
        if len(str(dt_like)) > 10:
            dt = datetime.fromtimestamp(dt_like / 10 ** (len(str(dt_like)) - 10), timezone.utc)
        else:
            dt = datetime.fromtimestamp(dt_like, timezone.utc)
    elif isinstance(dt_like, str):
        dt = dateparser.parse(dt_like, **kwargs)
    elif isinstance(dt_like, pd.Timestamp):
        dt = dt_like.to_pydatetime()
    elif isinstance(dt_like, np.datetime64):
        dt = datetime.combine(dt_like.astype(datetime), time())
    else:
        dt = dt_like

    if dt is None:
        raise ValueError("Couldn't parse the datetime")

    if not is_tz_aware(dt):
        dt = dt.replace(tzinfo=to_timezone(naive_tz))
    else:
        dt = dt.replace(tzinfo=to_timezone(dt.tzinfo))
    if tz is not None:
        dt = dt.astimezone(to_timezone(tz))
    return dt


def to_naive_datetime(dt: datetime) -> datetime:
    """Return the timezone info from a datetime."""
    return dt.astimezone().replace(tzinfo=None)


def datetime_to_ms(dt: datetime) -> int:
    """Convert a datetime to milliseconds."""
    epoch = datetime.fromtimestamp(0, dt.tzinfo)
    return int((dt - epoch).total_seconds() * 1000.0)


def interval_to_ms(interval: str) -> tp.Optional[int]:
    """Convert an interval string to milliseconds."""
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    try:
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000
    except (ValueError, KeyError):
        return None
