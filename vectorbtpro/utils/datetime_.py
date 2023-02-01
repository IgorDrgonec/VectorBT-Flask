# Copyright (c) 2023 Oleg Polakow. All rights reserved.

"""Utilities for working with dates and time."""

from datetime import datetime, timezone, timedelta, tzinfo, time
from dateutil.parser import parse

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import re
try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

from vectorbtpro import _typing as tp

__all__ = []

PandasDatetimeIndex = (pd.DatetimeIndex, pd.PeriodIndex)


def split_freq_str(freq: str) -> tp.Optional[tp.Tuple[int, str]]:
    """Split (human-readable) frequency into multiplier and unambiguous unit.

    Can be used both as offset and timedelta.

    The following units are returned:
    * "s" for second
    * "t" for minute
    * "h" for hour
    * "d" for day
    * "W" for week
    * "M" for month
    * "Q" for quarter
    * "Y" for year"""
    freq = "".join(freq.strip().split())
    match = re.match(r"^(\d*)\s*([a-zA-Z-]+)$", freq)
    if match.group(1) == "" and match.group(2).isnumeric():
        raise ValueError("Frequency must contain unit")
    if not match:
        return None
    if match.group(1) == "":
        multiplier = 1
    else:
        multiplier = int(match.group(1))
    if match.group(2) == "":
        raise ValueError("Frequency must contain unit")
    else:
        unit = match.group(2)
    if unit in ("S", "sec", "second", "seconds"):
        unit = "s"
    elif unit in ("T", "m", "min", "minute", "minutes"):
        unit = "t"
    elif unit in ("H", "hour", "hours", "hourly"):
        unit = "h"
    elif unit in ("D", "day", "days", "daily"):
        unit = "d"
    elif unit in ("w", "wk", "week", "weeks", "weekly"):
        unit = "W"
    elif unit in ("mo", "month", "months", "monthly"):
        unit = "M"
    elif unit in ("q", "quarter", "quarters", "quarterly"):
        unit = "Q"
    elif unit in ("y", "year", "years", "yearly", "annual", "annually"):
        unit = "Y"
    else:
        return None
    return multiplier, unit


def prepare_freq(freq: tp.FrequencyLike) -> tp.FrequencyLike:
    """Prepare frequency using `split_freq_str`.

    To include multiple units, separate them with comma."""
    if isinstance(freq, str):
        if ',' in freq:
            new_freq = ""
            for _freq in freq.split(','):
                split = split_freq_str(_freq)
                if split is not None:
                    new_freq += str(split[0]) + str(split[1])
                else:
                    return freq
            return new_freq
        split = split_freq_str(freq)
        if split is not None:
            freq = str(split[0]) + str(split[1])
        return freq
    return freq


def freq_to_timedelta(freq: tp.FrequencyLike) -> pd.Timedelta:
    """Convert a frequency-like object to `pd.Timedelta`."""
    if isinstance(freq, pd.Timedelta):
        return freq
    if isinstance(freq, str) and freq.startswith("-"):
        neg_td = True
        freq = freq[1:]
    else:
        neg_td = False
    freq = prepare_freq(freq)
    if isinstance(freq, str) and not freq[0].isdigit():
        # Otherwise "ValueError: unit abbreviation w/o a number"
        td = pd.Timedelta(1, unit=freq)
    else:
        td = pd.Timedelta(freq)
    if neg_td:
        return -td
    return td


def parse_timedelta(td: tp.TimedeltaLike) -> tp.Union[pd.Timedelta, pd.DateOffset]:
    """Parse a timedelta-like object into Pandas format."""
    if isinstance(td, (pd.Timedelta, pd.DateOffset)):
        return td
    try:
        return to_offset(td)
    finally:
        return freq_to_timedelta(td)


def time_to_timedelta(time: tp.TimeLike) -> pd.Timedelta:
    """Convert a time-like object into `pd.Timedelta`."""
    if isinstance(time, str):
        time = parse(time).time()
    return pd.Timedelta(
        hours=time.hour,
        minutes=time.minute,
        seconds=time.second,
        milliseconds=time.microsecond // 1000,
        microseconds=time.microsecond % 1000
    )


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
    """Try converting an index to a datetime index.

    Keyword arguments are passed to `pd.to_datetime`."""
    import dateparser
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if not isinstance(index, pd.Index):
        if isinstance(index, str):
            try:
                index = pd.to_datetime(index, **kwargs)
                index = [index]
            except Exception as e:
                if datetime_cfg["parse_index"]:
                    try:
                        parsed_index = dateparser.parse(index)
                        if parsed_index is None:
                            raise Exception
                        index = pd.to_datetime(parsed_index, **kwargs)
                        index = [index]
                    except Exception as e2:
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
            if datetime_cfg["parse_index"]:
                try:
                    def _parse(x):
                        _parsed_index = dateparser.parse(x)
                        if _parsed_index is None:
                            raise Exception
                        return _parsed_index
                    return pd.to_datetime(index.map(_parse), **kwargs)
                except Exception as e2:
                    pass
    return index


def try_align_to_datetime_index(source_index: tp.IndexLike, target_index: tp.Index, **kwargs) -> tp.Index:
    """Try aligning an index to another datetime index.

    Keyword arguments are passed to `try_to_datetime_index`."""
    source_index = try_to_datetime_index(source_index, **kwargs)
    if isinstance(source_index, pd.DatetimeIndex) and isinstance(target_index, pd.DatetimeIndex):
        if source_index.tzinfo is None and target_index.tzinfo is not None:
            source_index = source_index.tz_localize(target_index.tzinfo)
        elif source_index.tzinfo is not None and target_index.tzinfo is not None:
            source_index = source_index.tz_convert(target_index.tzinfo)
    return source_index


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


def to_timezone(tz: tp.TimezoneLike, to_fixed_offset: tp.Optional[bool] = None, **kwargs) -> tzinfo:
    """Parse the timezone.

    Strings are parsed by `zoneinfo` and `dateparser`, while integers and floats are treated as hour offsets.

    If `to_fixed_offset` is set to True, will convert to `datetime.timezone`. See global settings.

    `**kwargs` are passed to `dateparser.parse`."""
    import dateparser
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if tz is None:
        return get_local_tz()
    if to_fixed_offset is None:
        to_fixed_offset = datetime_cfg["to_fixed_offset"]

    if isinstance(tz, str):
        try:
            tz = zoneinfo.ZoneInfo(tz)
        except zoneinfo.ZoneInfoNotFoundError:
            dt = dateparser.parse("now %s" % tz, **kwargs)
            if dt is not None:
                tz = dt.tzinfo
    if isinstance(tz, (int, float)):
        tz = timezone(timedelta(hours=tz))
    if isinstance(tz, timedelta):
        tz = timezone(tz)
    if isinstance(tz, tzinfo):
        if to_fixed_offset:
            return timezone(tz.utcoffset(datetime.now()))
        return tz
    raise TypeError("Couldn't parse the timezone")


def to_datetime(dt_like: tp.DatetimeLike, **kwargs) -> datetime:
    """Parse the datetime as a `datetime.datetime`.

    See [dateparser docs](http://dateparser.readthedocs.io/en/latest/) for valid string formats and `**kwargs`."""
    if isinstance(dt_like, pd.Timestamp):
        return dt_like
    if isinstance(dt_like, (int, float)):
        return pd.to_datetime(dt_like, utc=True).to_pydatetime()
    if isinstance(dt_like, str):
        try:
            return pd.to_datetime(dt_like, **kwargs).to_pydatetime()
        except Exception as e:
            import dateparser

            dt = dateparser.parse(dt_like, **kwargs)
            if is_tz_aware(dt):
                dt = dt.replace(tzinfo=to_timezone(dt.tzinfo, to_fixed_offset=True))
            return dt
    return pd.to_datetime(dt_like, **kwargs).to_pydatetime()


def to_tzaware_datetime(
    dt_like: tp.DatetimeLike,
    naive_tz: tp.Optional[tp.TimezoneLike] = None,
    tz: tp.Optional[tp.TimezoneLike] = None,
    **kwargs
) -> datetime:
    """Parse the datetime as a timezone-aware `datetime.datetime`.

    Uses `to_datetime`.

    Raw timestamps are localized to UTC, while naive datetime is localized to `naive_tz`.
    Set `naive_tz` to None to use the default value defined under `vectorbtpro._settings.datetime`.
    To explicitly convert the datetime to a timezone, use `tz` (uses `to_timezone`)."""
    from vectorbtpro._settings import settings

    datetime_cfg = settings["datetime"]

    if naive_tz is None:
        naive_tz = datetime_cfg["naive_tz"]
    dt = to_datetime(dt_like, **kwargs)

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
