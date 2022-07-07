# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Generic Numba-compiled functions for records."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.generic.enums import *
from vectorbtpro.generic.nb.base import repartition_nb
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.template import Rep

# ############# Ranges ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1), gap_value=None),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_ranges_nb(arr: tp.Array2d, gap_value: tp.Scalar) -> tp.RecordArray:
    """Fill range records between gaps.

    Usage:
        * Find ranges in time series:

        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.generic.nb import get_ranges_nb

        >>> a = np.asarray([
        ...     [np.nan, np.nan, np.nan, np.nan],
        ...     [     2, np.nan, np.nan, np.nan],
        ...     [     3,      3, np.nan, np.nan],
        ...     [np.nan,      4,      4, np.nan],
        ...     [     5, np.nan,      5,      5],
        ...     [     6,      6, np.nan,      6]
        ... ])
        >>> records = get_ranges_nb(a, np.nan)

        >>> pd.DataFrame.from_records(records)
           id  col  start_idx  end_idx  status
        0   0    0          1        3       1
        1   1    0          4        5       0
        2   0    1          2        4       1
        3   1    1          5        5       0
        4   0    2          3        5       1
        5   0    3          4        5       0
        ```
    """
    new_records = np.empty(arr.shape, dtype=range_dt)
    counts = np.full(arr.shape[1], 0, dtype=np.int_)

    for col in prange(arr.shape[1]):
        range_started = False
        start_idx = -1
        end_idx = -1
        store_record = False
        status = -1

        for i in range(arr.shape[0]):
            cur_val = arr[i, col]

            if cur_val == gap_value or np.isnan(cur_val) and np.isnan(gap_value):
                if range_started:
                    # If stopped, save the current range
                    end_idx = i
                    range_started = False
                    store_record = True
                    status = RangeStatus.Closed
            else:
                if not range_started:
                    # If started, register a new range
                    start_idx = i
                    range_started = True

            if i == arr.shape[0] - 1 and range_started:
                # If still running, mark for save
                end_idx = arr.shape[0] - 1
                range_started = False
                store_record = True
                status = RangeStatus.Open

            if store_record:
                # Save range to the records
                r = counts[col]
                new_records["id"][r, col] = r
                new_records["col"][r, col] = col
                new_records["start_idx"][r, col] = start_idx
                new_records["end_idx"][r, col] = end_idx
                new_records["status"][r, col] = status
                counts[col] += 1

                # Reset running vars for a new range
                store_record = False

    return repartition_nb(new_records, counts)


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        n_rows=None,
        idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        id_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index=None,
        delta=None,
        delta_use_index=None,
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_ranges_from_delta_nb(
    n_rows: int,
    idx_arr: tp.Array1d,
    id_arr: tp.Array1d,
    col_map: tp.GroupMap,
    index: tp.Optional[tp.Array1d] = None,
    delta: int = 0,
    delta_use_index: bool = False,
) -> tp.RecordArray:
    """Build delta ranges."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.empty(idx_arr.shape[0], dtype=range_dt)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx: col_start_idx + col_len]

        for r in ridxs:
            if delta >= 0:
                start_idx = idx_arr[r]
                if delta_use_index:
                    if index is None:
                        raise ValueError("Index is required")
                    end_idx = len(index) - 1
                    status = RangeStatus.Open
                    for i in range(start_idx, index.shape[0]):
                        if index[i] >= index[start_idx] + delta:
                            end_idx = i
                            status = RangeStatus.Closed
                            break
                else:
                    if start_idx + delta < n_rows:
                        end_idx = start_idx + delta
                        status = RangeStatus.Closed
                    else:
                        end_idx = n_rows - 1
                        status = RangeStatus.Open
            else:
                end_idx = idx_arr[r]
                status = RangeStatus.Closed
                if delta_use_index:
                    if index is None:
                        raise ValueError("Index is required")
                    start_idx = 0
                    for i in range(end_idx, -1, -1):
                        if index[i] <= index[end_idx] + delta:
                            start_idx = i
                            break
                else:
                    if end_idx + delta >= 0:
                        start_idx = end_idx + delta
                    else:
                        start_idx = 0

            out["id"][r] = id_arr[r]
            out["col"][r] = col
            out["start_idx"][r] = start_idx
            out["end_idx"][r] = end_idx
            out["status"][r] = status

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0),
        status_arr=ch.ArraySlicer(axis=0),
        freq=None,
    ),
    merge_func=base_ch.concat,
)
@register_jitted(cache=True, tags={"can_parallel"})
def range_duration_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    freq: int = 1,
) -> tp.Array1d:
    """Get duration of each range record."""
    out = np.empty(start_idx_arr.shape[0], dtype=np.int_)
    for r in prange(start_idx_arr.shape[0]):
        if status_arr[r] == RangeStatus.Open:
            out[r] = end_idx_arr[r] - start_idx_arr[r] + freq
        else:
            out[r] = end_idx_arr[r] - start_idx_arr[r]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        end_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        status_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index_lens=ch.ArraySlicer(axis=0),
        overlapping=None,
        normalize=None,
    ),
    merge_func=base_ch.concat,
)
@register_jitted(cache=True, tags={"can_parallel"})
def range_coverage_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    col_map: tp.GroupMap,
    index_lens: tp.Array1d,
    overlapping: bool = False,
    normalize: bool = False,
) -> tp.Array1d:
    """Get coverage of range records.

    Set `overlapping` to True to get the number of overlapping steps.
    Set `normalize` to True to get the number of steps in relation either to the total number of steps
    (when `overlapping=False`) or to the number of covered steps (when `overlapping=True`).
    """
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full(col_lens.shape[0], np.nan, dtype=np.float_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]
        temp = np.full(index_lens[col], 0, dtype=np.int_)
        for r in ridxs:
            if status_arr[r] == RangeStatus.Open:
                temp[start_idx_arr[r] : end_idx_arr[r] + 1] += 1
            else:
                temp[start_idx_arr[r] : end_idx_arr[r]] += 1
        if overlapping:
            if normalize:
                out[col] = np.sum(temp > 1) / np.sum(temp > 0)
            else:
                out[col] = np.sum(temp > 1)
        else:
            if normalize:
                out[col] = np.sum(temp > 0) / index_lens[col]
            else:
                out[col] = np.sum(temp > 0)
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        end_idx_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        status_arr=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        index_len=None,
    ),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def ranges_to_mask_nb(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    col_map: tp.GroupMap,
    index_len: int,
) -> tp.Array2d:
    """Convert ranges to 2-dim mask."""
    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    out = np.full((index_len, col_lens.shape[0]), False, dtype=np.bool_)

    for col in prange(col_lens.shape[0]):
        col_len = col_lens[col]
        if col_len == 0:
            continue
        col_start_idx = col_start_idxs[col]
        ridxs = col_idxs[col_start_idx : col_start_idx + col_len]
        for r in ridxs:
            if status_arr[r] == RangeStatus.Open:
                out[start_idx_arr[r] : end_idx_arr[r] + 1, col] = True
            else:
                out[start_idx_arr[r] : end_idx_arr[r], col] = True

    return out


@register_jitted(cache=True)
def map_ranges_to_projections_nb(
    close: tp.Array2d,
    col_arr: tp.Array1d,
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array1d,
    index: tp.Optional[tp.Array1d] = None,
    proj_start: int = 0,
    proj_start_use_index: bool = False,
    proj_period: tp.Optional[int] = None,
    proj_period_use_index: bool = False,
    stretch: bool = False,
    normalize: bool = True,
    start_value: float = 1.0,
    ffill: bool = False,
    remove_empty: bool = False,
) -> tp.Tuple[tp.Array1d, tp.Array2d]:
    """Map each range into a projection.

    Returns two arrays:

    1. One-dimensional array where elements are record indices
    2. Two-dimensional array where rows are projections"""
    index_ranges_temp = np.empty((start_idx_arr.shape[0], 2), dtype=np.int_)

    max_duration = 0
    for r in range(start_idx_arr.shape[0]):
        if proj_start_use_index:
            if index is None:
                raise ValueError("Index is required")
            r_proj_start = len(index) - start_idx_arr[r]
            for i in range(start_idx_arr[r], index.shape[0]):
                if index[i] >= index[start_idx_arr[r]] + proj_start:
                    r_proj_start = i - start_idx_arr[r]
                    break
            r_start_idx = start_idx_arr[r] + r_proj_start
        else:
            r_start_idx = start_idx_arr[r] + proj_start
        if status_arr[r] == RangeStatus.Open:
            r_duration = end_idx_arr[r] - start_idx_arr[r] + 1
        else:
            r_duration = end_idx_arr[r] - start_idx_arr[r]
        if proj_period is None:
            r_end_idx = start_idx_arr[r] + r_duration
        else:
            if proj_period_use_index:
                if index is None:
                    raise ValueError("Index is required")
                r_proj_period = -1
                for i in range(r_start_idx, index.shape[0]):
                    if index[i] <= index[r_start_idx] + proj_period:
                        r_proj_period = i - r_start_idx
                    else:
                        break
            else:
                r_proj_period = proj_period
            if stretch:
                r_end_idx = r_start_idx + r_proj_period
            else:
                r_end_idx = min(start_idx_arr[r] + r_duration, r_start_idx + r_proj_period)
        r_end_idx = r_end_idx + 1
        if r_end_idx > close.shape[0]:
            r_end_idx = close.shape[0]
        if r_start_idx > r_end_idx:
            r_start_idx = r_end_idx
        if r_end_idx - r_start_idx > max_duration:
            max_duration = r_end_idx - r_start_idx
        index_ranges_temp[r, 0] = r_start_idx
        index_ranges_temp[r, 1] = r_end_idx

    ridx_out = np.empty((start_idx_arr.shape[0],), dtype=np.int_)
    proj_out = np.empty((start_idx_arr.shape[0], max_duration), dtype=np.float_)

    k = 0
    for r in range(start_idx_arr.shape[0]):
        if stretch:
            r_start_idx = index_ranges_temp[r, 0]
            r_end_idx = index_ranges_temp[r, 0] + proj_out.shape[1]
        else:
            r_start_idx = index_ranges_temp[r, 0]
            r_end_idx = index_ranges_temp[r, 1]
        r_close = close[r_start_idx:r_end_idx, col_arr[r]]
        any_set = False
        for i in range(proj_out.shape[1]):
            if i >= r_close.shape[0]:
                proj_out[k, i] = np.nan
            else:
                if normalize:
                    if i == 0:
                        proj_out[k, i] = start_value
                    else:
                        proj_out[k, i] = proj_out[k, i - 1] * r_close[i] / r_close[i - 1]
                else:
                    proj_out[k, i] = r_close[i]
            if not np.isnan(proj_out[k, i]) and i > 0:
                any_set = True
            if ffill and np.isnan(proj_out[k, i]) and i > 0:
                proj_out[k, i] = proj_out[k, i - 1]
        if any_set or not remove_empty:
            ridx_out[k] = r
            k += 1
    if remove_empty:
        return ridx_out[:k], proj_out[:k]
    return ridx_out, proj_out


# ############# Drawdowns ############# #


@register_jitted(cache=True)
def drawdown_1d_nb(arr: tp.Array1d) -> tp.Array1d:
    """Compute drawdown."""
    out = np.empty_like(arr, dtype=np.float_)
    max_val = np.nan
    for i in range(arr.shape[0]):
        if np.isnan(max_val) or arr[i] > max_val:
            max_val = arr[i]
        out[i] = arr[i] / max_val - 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(arr=ch.ArraySlicer(axis=1)),
    merge_func=base_ch.column_stack,
)
@register_jitted(cache=True, tags={"can_parallel"})
def drawdown_nb(arr: tp.Array2d) -> tp.Array2d:
    """2-dim version of `drawdown_1d_nb`."""
    out = np.empty_like(arr, dtype=np.float_)
    for col in prange(arr.shape[1]):
        out[:, col] = drawdown_1d_nb(arr[:, col])
    return out


@register_jitted(cache=True)
def fill_drawdown_record_nb(
    new_records: tp.RecordArray2d,
    counts: tp.Array2d,
    i: int,
    col: int,
    peak_idx: int,
    valley_idx: int,
    peak_val: float,
    valley_val: float,
    end_val: float,
    status: int,
):
    """Fill a drawdown record."""
    r = counts[col]
    new_records["id"][r, col] = r
    new_records["col"][r, col] = col
    new_records["peak_idx"][r, col] = peak_idx
    new_records["start_idx"][r, col] = peak_idx + 1
    new_records["valley_idx"][r, col] = valley_idx
    new_records["end_idx"][r, col] = i
    new_records["peak_val"][r, col] = peak_val
    new_records["valley_val"][r, col] = valley_val
    new_records["end_val"][r, col] = end_val
    new_records["status"][r, col] = status
    counts[col] += 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        open=ch.ArraySlicer(axis=1),
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
    ),
    merge_func=records_ch.merge_records,
    merge_kwargs=dict(chunk_meta=Rep("chunk_meta")),
)
@register_jitted(cache=True, tags={"can_parallel"})
def get_drawdowns_nb(
    open: tp.Optional[tp.Array2d],
    high: tp.Optional[tp.Array2d],
    low: tp.Optional[tp.Array2d],
    close: tp.Array2d,
) -> tp.RecordArray:
    """Fill drawdown records by analyzing a time series.

    Only `close` must be provided, other time series are optional.

    Usage:
        ```pycon
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbtpro.generic.nb import get_drawdowns_nb

        >>> close = np.asarray([
        ...     [1, 5, 1, 3],
        ...     [2, 4, 2, 2],
        ...     [3, 3, 3, 1],
        ...     [4, 2, 2, 2],
        ...     [5, 1, 1, 3]
        ... ])
        >>> records = get_drawdowns_nb(None, None, None, close)

        >>> pd.DataFrame.from_records(records)
           id  col  peak_idx  start_idx  valley_idx  end_idx  peak_val  valley_val  \\
        0   0    1         0          1           4        4       5.0         1.0
        1   0    2         2          3           4        4       3.0         1.0
        2   0    3         0          1           2        4       3.0         1.0

           end_val  status
        0      1.0       0
        1      1.0       0
        2      3.0       1
        ```
    """
    new_records = np.empty(close.shape, dtype=drawdown_dt)
    counts = np.full(close.shape[1], 0, dtype=np.int_)

    for col in prange(close.shape[1]):
        drawdown_started = False
        _close = close[0, col]
        if open is None:
            _open = np.nan
        else:
            _open = open[0, col]
        peak_idx = 0
        valley_idx = 0
        peak_val = _open
        valley_val = _open

        for i in range(close.shape[0]):
            _close = close[i, col]
            if open is None:
                _open = np.nan
            else:
                _open = open[i, col]
            if high is None:
                _high = np.nan
            else:
                _high = high[i, col]
            if low is None:
                _low = np.nan
            else:
                _low = low[i, col]
            if np.isnan(_high):
                if np.isnan(_open):
                    _high = _close
                elif np.isnan(_close):
                    _high = _open
                else:
                    _high = max(_open, _close)
            if np.isnan(_low):
                if np.isnan(_open):
                    _low = _close
                elif np.isnan(_close):
                    _low = _open
                else:
                    _low = min(_open, _close)

            if drawdown_started:
                if _open >= peak_val:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        peak_idx=peak_idx,
                        valley_idx=valley_idx,
                        peak_val=peak_val,
                        valley_val=valley_val,
                        end_val=_open,
                        status=DrawdownStatus.Recovered,
                    )
                    peak_idx = i
                    valley_idx = i
                    peak_val = _open
                    valley_val = _open

            if drawdown_started:
                if _low < valley_val:
                    valley_idx = i
                    valley_val = _low
                if _high >= peak_val:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        peak_idx=peak_idx,
                        valley_idx=valley_idx,
                        peak_val=peak_val,
                        valley_val=valley_val,
                        end_val=_high,
                        status=DrawdownStatus.Recovered,
                    )
                    peak_idx = i
                    valley_idx = i
                    peak_val = _high
                    valley_val = _high
            else:
                if np.isnan(peak_val) or _high >= peak_val:
                    peak_idx = i
                    valley_idx = i
                    peak_val = _high
                    valley_val = _high
                elif _low < valley_val:
                    if not np.isnan(valley_val):
                        drawdown_started = True
                    valley_idx = i
                    valley_val = _low

            if drawdown_started:
                if i == close.shape[0] - 1:
                    drawdown_started = False
                    fill_drawdown_record_nb(
                        new_records=new_records,
                        counts=counts,
                        i=i,
                        col=col,
                        peak_idx=peak_idx,
                        valley_idx=valley_idx,
                        peak_val=peak_val,
                        valley_val=valley_val,
                        end_val=_close,
                        status=DrawdownStatus.Active,
                    )

    return repartition_nb(new_records, counts)


@register_chunkable(
    size=ch.ArraySizer(arg_query="peak_val_arr", axis=0),
    arg_take_spec=dict(peak_val_arr=ch.ArraySlicer(axis=0), valley_val_arr=ch.ArraySlicer(axis=0)),
    merge_func=base_ch.concat,
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_drawdown_nb(peak_val_arr: tp.Array1d, valley_val_arr: tp.Array1d) -> tp.Array1d:
    """Compute the drawdown of each drawdown record."""
    out = np.empty(valley_val_arr.shape[0], dtype=np.float_)
    for r in prange(valley_val_arr.shape[0]):
        out[r] = (valley_val_arr[r] - peak_val_arr[r]) / peak_val_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(start_idx_arr=ch.ArraySlicer(axis=0), valley_idx_arr=ch.ArraySlicer(axis=0)),
    merge_func=base_ch.concat,
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_decline_duration_nb(start_idx_arr: tp.Array1d, valley_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the duration of the peak-to-valley phase of each drawdown record."""
    out = np.empty(valley_idx_arr.shape[0], dtype=np.float_)
    for r in prange(valley_idx_arr.shape[0]):
        out[r] = valley_idx_arr[r] - start_idx_arr[r] + 1
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="valley_idx_arr", axis=0),
    arg_take_spec=dict(valley_idx_arr=ch.ArraySlicer(axis=0), end_idx_arr=ch.ArraySlicer(axis=0)),
    merge_func=base_ch.concat,
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_duration_nb(valley_idx_arr: tp.Array1d, end_idx_arr: tp.Array1d) -> tp.Array1d:
    """Compute the duration of the valley-to-recovery phase of each drawdown record."""
    out = np.empty(end_idx_arr.shape[0], dtype=np.float_)
    for r in prange(end_idx_arr.shape[0]):
        out[r] = end_idx_arr[r] - valley_idx_arr[r]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="start_idx_arr", axis=0),
    arg_take_spec=dict(
        start_idx_arr=ch.ArraySlicer(axis=0),
        valley_idx_arr=ch.ArraySlicer(axis=0),
        end_idx_arr=ch.ArraySlicer(axis=0),
    ),
    merge_func=base_ch.concat,
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_duration_ratio_nb(
    start_idx_arr: tp.Array1d,
    valley_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
) -> tp.Array1d:
    """Compute the ratio of the recovery duration to the decline duration of each drawdown record."""
    out = np.empty(start_idx_arr.shape[0], dtype=np.float_)
    for r in prange(start_idx_arr.shape[0]):
        out[r] = (end_idx_arr[r] - valley_idx_arr[r]) / (valley_idx_arr[r] - start_idx_arr[r] + 1)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="valley_val_arr", axis=0),
    arg_take_spec=dict(valley_val_arr=ch.ArraySlicer(axis=0), end_val_arr=ch.ArraySlicer(axis=0)),
    merge_func=base_ch.concat,
)
@register_jitted(cache=True, tags={"can_parallel"})
def dd_recovery_return_nb(valley_val_arr: tp.Array1d, end_val_arr: tp.Array1d) -> tp.Array1d:
    """Compute the recovery return of each drawdown record."""
    out = np.empty(end_val_arr.shape[0], dtype=np.float_)
    for r in prange(end_val_arr.shape[0]):
        out[r] = (end_val_arr[r] - valley_val_arr[r]) / valley_val_arr[r]
    return out
