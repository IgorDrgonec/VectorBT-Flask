import os
from datetime import datetime

import pytest
from numba import njit

import vectorbtpro as vbt
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.enums import *

from tests.utils import *

seed = 42

day_dt = np.timedelta64(86400000000000)

price = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    index=pd.Index(
        [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4), datetime(2020, 1, 5)],
    ),
)
price_wide = price.vbt.tile(3, keys=["a", "b", "c"])


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# from_signals ############# #

entries = pd.Series([True, True, True, False, False], index=price.index)
entries_wide = entries.vbt.tile(3, keys=["a", "b", "c"])

exits = pd.Series([False, False, True, True, True], index=price.index)
exits_wide = exits.vbt.tile(3, keys=["a", "b", "c"])


def from_signals_both(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, entries, exits, direction="both", **kwargs)


def from_signals_longonly(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, entries, exits, direction="longonly", **kwargs)


def from_signals_shortonly(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, entries, exits, direction="shortonly", **kwargs)


def from_ls_signals_both(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, entries, False, exits, False, **kwargs)


def from_ls_signals_longonly(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, entries, exits, False, False, **kwargs)


def from_ls_signals_shortonly(close=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(close, False, False, entries, exits, **kwargs)


class TestFromSignals:
    def test_data(self):
        data = vbt.RandomOHLCData.fetch(
            [0, 1],
            ohlc_freq="1d",
            start="2020-01-01",
            end="2020-02-01",
            freq="1h",
            seed=42,
        )
        pf = vbt.Portfolio.from_signals(data)
        assert pf.open is not None
        assert pf.high is not None
        assert pf.low is not None
        assert pf.close is not None
        pf = vbt.Portfolio.from_signals(data.get("Close"))
        assert pf.open is None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_signals(data[["Open", "Close"]])
        assert pf.open is not None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_signals(data["Close"])
        assert pf.open is None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None
        pf = vbt.Portfolio.from_signals(data["Close"], open=data.get("Open"))
        assert pf.open is not None
        assert pf.high is None
        assert pf.low is None
        assert pf.close is not None

    @pytest.mark.parametrize(
        "test_ls",
        [False, True],
    )
    def test_one_column(self, test_ls):
        _from_signals_both = from_ls_signals_both if test_ls else from_signals_both
        _from_signals_longonly = from_ls_signals_longonly if test_ls else from_signals_longonly
        _from_signals_shortonly = from_ls_signals_shortonly if test_ls else from_signals_shortonly
        assert_records_close(
            _from_signals_both().order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 3, 200.0, 4.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            _from_signals_longonly().order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 3, 100.0, 4.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            _from_signals_shortonly().order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 0, 3, 50.0, 4.0, 0.0, 0)], dtype=order_dt),
        )
        pf = _from_signals_both()
        assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        assert_index_equal(pf.wrapper.columns, pd.Index([0], dtype="int64"))
        assert pf.wrapper.ndim == 1
        assert pf.wrapper.freq == day_dt
        assert pf.wrapper.grouper.group_by is None

    @pytest.mark.parametrize(
        "test_ls",
        [False, True],
    )
    def test_multiple_columns(self, test_ls):
        _from_signals_both = from_ls_signals_both if test_ls else from_signals_both
        _from_signals_longonly = from_ls_signals_longonly if test_ls else from_signals_longonly
        _from_signals_shortonly = from_ls_signals_shortonly if test_ls else from_signals_shortonly
        assert_records_close(
            _from_signals_both(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 200.0, 4.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 3, 200.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            _from_signals_longonly(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 100.0, 4.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 3, 100.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 100.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            _from_signals_shortonly(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (1, 0, 3, 50.0, 4.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (1, 1, 3, 50.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 3, 50.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pf = _from_signals_both(close=price_wide)
        assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        assert_index_equal(pf.wrapper.columns, pd.Index(["a", "b", "c"], dtype="object"))
        assert pf.wrapper.ndim == 2
        assert pf.wrapper.freq == day_dt
        assert pf.wrapper.grouper.group_by is None

    def test_custom_signal_func(self):
        @njit
        def signal_func_nb(c, long_num_arr, short_num_arr):
            long_num = nb.get_elem_nb(c, long_num_arr)
            short_num = nb.get_elem_nb(c, short_num_arr)
            is_long_entry = long_num > 0
            is_long_exit = long_num < 0
            is_short_entry = short_num > 0
            is_short_exit = short_num < 0
            return is_long_entry, is_long_exit, is_short_entry, is_short_exit

        pf_base = vbt.Portfolio.from_signals(
            pd.Series([1, 2, 3, 4, 5]),
            entries=pd.Series([True, False, False, False, False]),
            exits=pd.Series([False, False, True, False, False]),
            short_entries=pd.Series([False, True, False, True, False]),
            short_exits=pd.Series([False, False, False, False, True]),
            size=1,
            upon_opposite_entry="ignore",
        )
        pf = vbt.Portfolio.from_signals(
            pd.Series([1, 2, 3, 4, 5]),
            signal_func_nb=signal_func_nb,
            signal_args=(vbt.Rep("long_num_arr"), vbt.Rep("short_num_arr")),
            broadcast_named_args=dict(
                long_num_arr=pd.Series([1, 0, -1, 0, 0]),
                short_num_arr=pd.Series([0, 1, 0, 1, -1]),
            ),
            size=1,
            upon_opposite_entry="ignore",
        )
        assert_records_close(pf_base.order_records, pf.order_records)

    def test_amount(self):
        assert_records_close(
            from_signals_both(size=[[0, 1, np.inf]], size_type="amount").order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 2.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=[[0, 1, np.inf]], size_type="amount").order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 100.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=[[0, 1, np.inf]], size_type="amount").order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 3, 50.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_value(self):
        assert_records_close(
            from_signals_both(size=[[0, 1, np.inf]], size_type="value").order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.25, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=[[0, 1, np.inf]], size_type="value").order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 100.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=[[0, 1, np.inf]], size_type="value").order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 3, 50.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_percent(self):
        with pytest.raises(Exception):
            from_signals_both(size=0.5, size_type="percent")
        assert_records_close(
            from_signals_both(size=0.5, size_type="percent", upon_opposite_entry="close").order_records,
            np.array(
                [(0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 0, 3, 50.0, 4.0, 0.0, 1), (2, 0, 4, 25.0, 5.0, 0.0, 1)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                size=0.5,
                size_type="percent",
                upon_opposite_entry="close",
                accumulate=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 0),
                    (2, 0, 3, 62.5, 4.0, 0.0, 1),
                    (3, 0, 4, 27.5, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=0.5, size_type="percent").order_records,
            np.array([(0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 0, 3, 50.0, 4.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_shortonly(size=0.5, size_type="percent").order_records,
            np.array([(0, 0, 0, 50.0, 1.0, 0.0, 1), (1, 0, 3, 37.5, 4.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_longonly(
                close=price_wide,
                size=0.5,
                size_type="percent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 3, 50.0, 4.0, 0.0, 1),
                    (0, 1, 0, 25.0, 1.0, 0.0, 0),
                    (1, 1, 3, 25.0, 4.0, 0.0, 1),
                    (0, 2, 0, 12.5, 1.0, 0.0, 0),
                    (1, 2, 3, 12.5, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_price(self):
        assert_records_close(
            from_signals_both(price=price * 1.01).order_records,
            np.array(
                [(0, 0, 0, 99.00990099009901, 1.01, 0.0, 0), (1, 0, 3, 198.01980198019803, 4.04, 0.0, 1)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(price=price * 1.01).order_records,
            np.array([(0, 0, 0, 99.00990099, 1.01, 0.0, 0), (1, 0, 3, 99.00990099, 4.04, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_shortonly(price=price * 1.01).order_records,
            np.array(
                [(0, 0, 0, 99.00990099009901, 1.01, 0.0, 1), (1, 0, 3, 49.504950495049506, 4.04, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(price=np.inf).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 3, 200.0, 4.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_longonly(price=np.inf).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 3, 100.0, 4.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_shortonly(price=np.inf).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 0, 3, 50.0, 4.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_both(price=-np.inf, open=price.shift(1)).order_records,
            np.array([(0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 0, 3, 200.0, 3.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_longonly(price=-np.inf, open=price.shift(1)).order_records,
            np.array([(0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 0, 3, 100.0, 3.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_shortonly(price=-np.inf, open=price.shift(1)).order_records,
            np.array([(0, 0, 1, 100.0, 1.0, 0.0, 1), (1, 0, 3, 66.66666666666667, 3.0, 0.0, 0)], dtype=order_dt),
        )

    def test_price_area(self):
        assert_records_close(
            from_signals_both(
                open=2,
                high=4,
                low=1,
                close=3,
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [(0, 0, 0, 1.0, 0.55, 0.0, 0), (0, 1, 0, 1.0, 3.3, 0.0, 0), (0, 2, 0, 1.0, 5.5, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [(0, 0, 0, 1.0, 0.55, 0.0, 0), (0, 1, 0, 1.0, 3.3, 0.0, 0), (0, 2, 0, 1.0, 5.5, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [(0, 0, 0, 1.0, 0.45, 0.0, 1), (0, 1, 0, 1.0, 2.7, 0.0, 1), (0, 2, 0, 1.0, 4.5, 0.0, 1)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [(0, 0, 0, 1.0, 1.0, 0.0, 0), (0, 1, 0, 1.0, 3.0, 0.0, 0), (0, 2, 0, 1.0, 4.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [(0, 0, 0, 1.0, 1.0, 0.0, 0), (0, 1, 0, 1.0, 3.0, 0.0, 0), (0, 2, 0, 1.0, 4.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
                entries=True,
                exits=False,
                price=[[0.5, np.inf, 5]],
                size=1,
                slippage=0.1,
            ).order_records,
            np.array(
                [(0, 0, 0, 1.0, 1.0, 0.0, 1), (0, 1, 0, 1.0, 3.0, 0.0, 1), (0, 2, 0, 1.0, 4.0, 0.0, 1)],
                dtype=order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_longonly(
                entries=True,
                exits=False,
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=0.5,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=np.inf,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=5,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=0.5,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=np.inf,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_signals_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                entries=True,
                exits=False,
                price=5,
                size=1,
                slippage=0.1,
            )

    def test_val_price(self):
        price_nan = pd.Series([1, 2, np.nan, 4, 5], index=price.index)
        assert_records_close(
            from_signals_both(close=price_nan, size=1, val_price=np.inf, size_type="value").order_records,
            from_signals_both(close=price_nan, size=1, val_price=price, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_longonly(close=price_nan, size=1, val_price=np.inf, size_type="value").order_records,
            from_signals_longonly(close=price_nan, size=1, val_price=price, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_shortonly(close=price_nan, size=1, val_price=np.inf, size_type="value").order_records,
            from_signals_shortonly(close=price_nan, size=1, val_price=price, size_type="value").order_records,
        )
        shift_price = price_nan.ffill().shift(1)
        assert_records_close(
            from_signals_longonly(close=price_nan, size=1, val_price=-np.inf, size_type="value").order_records,
            from_signals_longonly(close=price_nan, size=1, val_price=shift_price, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_longonly(close=price_nan, size=1, val_price=-np.inf, size_type="value").order_records,
            from_signals_longonly(close=price_nan, size=1, val_price=shift_price, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_shortonly(close=price_nan, size=1, val_price=-np.inf, size_type="value").order_records,
            from_signals_shortonly(close=price_nan, size=1, val_price=shift_price, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_both(
                close=price_nan,
                size=1,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_both(
                close=price_nan,
                size=1,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan,
                size=1,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_longonly(
                close=price_nan,
                size=1,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan,
                size=1,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_shortonly(
                close=price_nan,
                size=1,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        price_all_nan = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=price.index)
        assert_records_close(
            from_signals_both(
                close=price_nan,
                size=1,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_both(
                close=price_nan,
                size=1,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan,
                size=1,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_longonly(
                close=price_nan,
                size=1,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan,
                size=1,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_signals_shortonly(
                close=price_nan,
                size=1,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_signals_both(close=price_nan, size=1, val_price=np.nan, size_type="value").order_records,
            from_signals_both(close=price_nan, size=1, val_price=shift_price, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_longonly(close=price_nan, size=1, val_price=np.nan, size_type="value").order_records,
            from_signals_longonly(close=price_nan, size=1, val_price=shift_price, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_shortonly(close=price_nan, size=1, val_price=np.nan, size_type="value").order_records,
            from_signals_shortonly(close=price_nan, size=1, val_price=shift_price, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_both(
                close=price_nan,
                open=price_nan,
                size=1,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_signals_both(close=price_nan, size=1, val_price=price_nan, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=price_nan,
                open=price_nan,
                size=1,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_signals_longonly(close=price_nan, size=1, val_price=price_nan, size_type="value").order_records,
        )
        assert_records_close(
            from_signals_shortonly(
                close=price_nan,
                open=price_nan,
                size=1,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_signals_shortonly(close=price_nan, size=1, val_price=price_nan, size_type="value").order_records,
        )

    def test_fees(self):
        assert_records_close(
            from_signals_both(size=1, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 3, 2.0, 4.0, -0.8, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 2.0, 4.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 3, 2.0, 4.0, 0.8, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 3, 2.0, 4.0, 8.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 3, 1.0, 4.0, -0.4, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 3, 1.0, 4.0, 0.4, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 3, 1.0, 4.0, 4.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 1),
                    (1, 0, 3, 1.0, 4.0, -0.4, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.1, 1),
                    (1, 2, 3, 1.0, 4.0, 0.4, 0),
                    (0, 3, 0, 1.0, 1.0, 1.0, 1),
                    (1, 3, 3, 1.0, 4.0, 4.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_fixed_fees(self):
        assert_records_close(
            from_signals_both(size=1, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 3, 2.0, 4.0, -0.1, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 2.0, 4.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 3, 2.0, 4.0, 0.1, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 3, 2.0, 4.0, 1.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 3, 1.0, 4.0, -0.1, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 3, 1.0, 4.0, 0.1, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 3, 1.0, 4.0, 1.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 1),
                    (1, 0, 3, 1.0, 4.0, -0.1, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.1, 1),
                    (1, 2, 3, 1.0, 4.0, 0.1, 0),
                    (0, 3, 0, 1.0, 1.0, 1.0, 1),
                    (1, 3, 3, 1.0, 4.0, 1.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_slippage(self):
        assert_records_close(
            from_signals_both(size=1, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 3, 2.0, 4.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.1, 0.0, 0),
                    (1, 1, 3, 2.0, 3.6, 0.0, 1),
                    (0, 2, 0, 1.0, 2.0, 0.0, 0),
                    (1, 2, 3, 2.0, 0.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 3, 1.0, 4.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.1, 0.0, 0),
                    (1, 1, 3, 1.0, 3.6, 0.0, 1),
                    (0, 2, 0, 1.0, 2.0, 0.0, 0),
                    (1, 2, 3, 1.0, 0.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 3, 1.0, 4.0, 0.0, 0),
                    (0, 1, 0, 1.0, 0.9, 0.0, 1),
                    (1, 1, 3, 1.0, 4.4, 0.0, 0),
                    (0, 2, 0, 1.0, 0.0, 0.0, 1),
                    (1, 2, 3, 1.0, 8.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_min_size(self):
        assert_records_close(
            from_signals_both(size=1, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 3, 2.0, 4.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 2.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 3, 1.0, 4.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 3, 1.0, 4.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_max_size(self):
        assert_records_close(
            from_signals_both(size=1, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 0),
                    (1, 0, 3, 0.5, 4.0, 0.0, 1),
                    (2, 0, 4, 0.5, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                    (2, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 3, 2.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 0),
                    (1, 0, 3, 0.5, 4.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 3, 1.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 1),
                    (1, 0, 3, 0.5, 4.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.0, 1),
                    (1, 2, 3, 1.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_reject_prob(self):
        assert_records_close(
            from_signals_both(size=1.0, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 3, 2.0, 4.0, 0.0, 1),
                    (0, 1, 1, 1.0, 2.0, 0.0, 0),
                    (1, 1, 3, 2.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1.0, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 3, 1.0, 4.0, 0.0, 1),
                    (0, 1, 1, 1.0, 2.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1.0, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 3, 1.0, 4.0, 0.0, 0),
                    (0, 1, 1, 1.0, 2.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_allow_partial(self):
        assert_records_close(
            from_signals_both(size=1000, allow_partial=[[True, False]]).order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 3, 1100.0, 4.0, 0.0, 1), (0, 1, 3, 1000.0, 4.0, 0.0, 1)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1000, allow_partial=[[True, False]]).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 3, 100.0, 4.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_shortonly(size=1000, allow_partial=[[True, False]]).order_records,
            np.array(
                [(0, 0, 0, 1000.0, 1.0, 0.0, 1), (1, 0, 3, 275.0, 4.0, 0.0, 0), (0, 1, 0, 1000.0, 1.0, 0.0, 1)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 200.0, 4.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 100.0, 4.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 3, 100.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 0, 3, 50.0, 4.0, 0.0, 0), (0, 1, 0, 100.0, 1.0, 0.0, 1)],
                dtype=order_dt,
            ),
        )

    def test_raise_reject(self):
        assert_records_close(
            from_signals_both(size=1000, allow_partial=True, raise_reject=True).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 3, 1100.0, 4.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_signals_longonly(size=1000, allow_partial=True, raise_reject=True).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 3, 100.0, 4.0, 0.0, 1)], dtype=order_dt),
        )
        with pytest.raises(Exception):
            from_signals_shortonly(size=1000, allow_partial=True, raise_reject=True).order_records
        with pytest.raises(Exception):
            from_signals_both(size=1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception):
            from_signals_longonly(size=1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception):
            from_signals_shortonly(size=1000, allow_partial=False, raise_reject=True).order_records

    def test_log(self):
        assert_records_close(
            from_signals_both(log=True).log_records,
            np.array(
                [
                    (
                        0,
                        0,
                        0,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        100.0,
                        0.0,
                        0.0,
                        100.0,
                        1.0,
                        100.0,
                        np.inf,
                        np.inf,
                        0,
                        2,
                        0.0,
                        0.0,
                        0.0,
                        1e-08,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        0.0,
                        100.0,
                        0.0,
                        0.0,
                        1.0,
                        100.0,
                        100.0,
                        1.0,
                        0.0,
                        0,
                        0,
                        -1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        0.0,
                        100.0,
                        0.0,
                        0.0,
                        4.0,
                        400.0,
                        -np.inf,
                        np.inf,
                        0,
                        2,
                        0.0,
                        0.0,
                        0.0,
                        1e-08,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        800.0,
                        -100.0,
                        400.0,
                        0.0,
                        4.0,
                        400.0,
                        200.0,
                        4.0,
                        0.0,
                        1,
                        0,
                        -1,
                        1,
                    ),
                ],
                dtype=log_dt,
            ),
        )

    def test_accumulate(self):
        assert_records_close(
            from_signals_both(size=1, accumulate=[["disabled", "addonly", "removeonly", "both"]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 3, 2.0, 4.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 3.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 3, 1.0, 4.0, 0.0, 1),
                    (2, 2, 4, 1.0, 5.0, 0.0, 1),
                    (0, 3, 0, 1.0, 1.0, 0.0, 0),
                    (1, 3, 1, 1.0, 2.0, 0.0, 0),
                    (2, 3, 3, 1.0, 4.0, 0.0, 1),
                    (3, 3, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(size=1, accumulate=[["disabled", "addonly", "removeonly", "both"]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 3, 1.0, 4.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 2.0, 4.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 3, 1.0, 4.0, 0.0, 1),
                    (0, 3, 0, 1.0, 1.0, 0.0, 0),
                    (1, 3, 1, 1.0, 2.0, 0.0, 0),
                    (2, 3, 3, 1.0, 4.0, 0.0, 1),
                    (3, 3, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(size=1, accumulate=[["disabled", "addonly", "removeonly", "both"]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 3, 1.0, 4.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 2.0, 4.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.0, 1),
                    (1, 2, 3, 1.0, 4.0, 0.0, 0),
                    (0, 3, 0, 1.0, 1.0, 0.0, 1),
                    (1, 3, 1, 1.0, 2.0, 0.0, 1),
                    (2, 3, 3, 1.0, 4.0, 0.0, 0),
                    (3, 3, 4, 1.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_upon_long_conflict(self):
        kwargs = dict(
            close=price[:3],
            entries=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, True, False],
                    [True, True, True, True, True, True, True],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [False, False, False, False, True, False, True],
                    [True, True, True, True, True, True, True],
                ]
            ),
            size=1.0,
            accumulate=True,
            upon_long_conflict=[["ignore", "entry", "exit", "adjacent", "adjacent", "opposite", "opposite"]],
        )
        assert_records_close(
            from_signals_longonly(**kwargs).order_records,
            np.array(
                [
                    (0, 0, 1, 1.0, 2.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 2, 1.0, 3.0, 0.0, 0),
                    (0, 2, 1, 1.0, 2.0, 0.0, 0),
                    (1, 2, 2, 1.0, 3.0, 0.0, 1),
                    (0, 3, 1, 1.0, 2.0, 0.0, 0),
                    (1, 3, 2, 1.0, 3.0, 0.0, 0),
                    (0, 5, 1, 1.0, 2.0, 0.0, 0),
                    (1, 5, 2, 1.0, 3.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_upon_short_conflict(self):
        kwargs = dict(
            close=price[:3],
            entries=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, True, False],
                    [True, True, True, True, True, True, True],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [False, False, False, False, True, False, True],
                    [True, True, True, True, True, True, True],
                ]
            ),
            size=1.0,
            accumulate=True,
            upon_short_conflict=[["ignore", "entry", "exit", "adjacent", "adjacent", "opposite", "opposite"]],
        )
        assert_records_close(
            from_signals_shortonly(**kwargs).order_records,
            np.array(
                [
                    (0, 0, 1, 1.0, 2.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 2, 1.0, 3.0, 0.0, 1),
                    (0, 2, 1, 1.0, 2.0, 0.0, 1),
                    (1, 2, 2, 1.0, 3.0, 0.0, 0),
                    (0, 3, 1, 1.0, 2.0, 0.0, 1),
                    (1, 3, 2, 1.0, 3.0, 0.0, 1),
                    (0, 5, 1, 1.0, 2.0, 0.0, 1),
                    (1, 5, 2, 1.0, 3.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_upon_dir_conflict(self):
        kwargs = dict(
            close=price[:3],
            entries=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, True, False],
                    [True, True, True, True, True, True, True],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [True, True, True, True, True, True, True],
                    [False, False, False, False, True, False, True],
                    [True, True, True, True, True, True, True],
                ]
            ),
            size=1.0,
            accumulate=True,
            upon_dir_conflict=[["ignore", "long", "short", "adjacent", "adjacent", "opposite", "opposite"]],
        )
        assert_records_close(
            from_signals_both(**kwargs).order_records,
            np.array(
                [
                    (0, 0, 1, 1.0, 2.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 2, 1.0, 3.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.0, 1),
                    (1, 2, 1, 1.0, 2.0, 0.0, 0),
                    (2, 2, 2, 1.0, 3.0, 0.0, 1),
                    (0, 3, 1, 1.0, 2.0, 0.0, 0),
                    (1, 3, 2, 1.0, 3.0, 0.0, 0),
                    (0, 4, 1, 1.0, 2.0, 0.0, 1),
                    (1, 4, 2, 1.0, 3.0, 0.0, 1),
                    (0, 5, 1, 1.0, 2.0, 0.0, 0),
                    (1, 5, 2, 1.0, 3.0, 0.0, 1),
                    (0, 6, 1, 1.0, 2.0, 0.0, 1),
                    (1, 6, 2, 1.0, 3.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_upon_opposite_entry(self):
        kwargs = dict(
            close=price[:3],
            entries=pd.DataFrame(
                [
                    [True, False, True, False, True, False, True, False, True, False],
                    [False, True, False, True, False, True, False, True, False, True],
                    [True, False, True, False, True, False, True, False, True, False],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [False, True, False, True, False, True, False, True, False, True],
                    [True, False, True, False, True, False, True, False, True, False],
                    [False, True, False, True, False, True, False, True, False, True],
                ]
            ),
            size=1.0,
            upon_opposite_entry=[
                [
                    "ignore",
                    "ignore",
                    "close",
                    "close",
                    "closereduce",
                    "closereduce",
                    "reverse",
                    "reverse",
                    "reversereduce",
                    "reversereduce",
                ]
            ],
        )
        assert_records_close(
            from_signals_both(**kwargs).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 1, 1.0, 2.0, 0.0, 1),
                    (2, 2, 2, 1.0, 3.0, 0.0, 0),
                    (0, 3, 0, 1.0, 1.0, 0.0, 1),
                    (1, 3, 1, 1.0, 2.0, 0.0, 0),
                    (2, 3, 2, 1.0, 3.0, 0.0, 1),
                    (0, 4, 0, 1.0, 1.0, 0.0, 0),
                    (1, 4, 1, 1.0, 2.0, 0.0, 1),
                    (2, 4, 2, 1.0, 3.0, 0.0, 0),
                    (0, 5, 0, 1.0, 1.0, 0.0, 1),
                    (1, 5, 1, 1.0, 2.0, 0.0, 0),
                    (2, 5, 2, 1.0, 3.0, 0.0, 1),
                    (0, 6, 0, 1.0, 1.0, 0.0, 0),
                    (1, 6, 1, 2.0, 2.0, 0.0, 1),
                    (2, 6, 2, 2.0, 3.0, 0.0, 0),
                    (0, 7, 0, 1.0, 1.0, 0.0, 1),
                    (1, 7, 1, 2.0, 2.0, 0.0, 0),
                    (2, 7, 2, 2.0, 3.0, 0.0, 1),
                    (0, 8, 0, 1.0, 1.0, 0.0, 0),
                    (1, 8, 1, 2.0, 2.0, 0.0, 1),
                    (2, 8, 2, 2.0, 3.0, 0.0, 0),
                    (0, 9, 0, 1.0, 1.0, 0.0, 1),
                    (1, 9, 1, 2.0, 2.0, 0.0, 0),
                    (2, 9, 2, 2.0, 3.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(**kwargs, accumulate=True).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 2, 1.0, 3.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 2, 1.0, 3.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 1, 1.0, 2.0, 0.0, 1),
                    (2, 2, 2, 1.0, 3.0, 0.0, 0),
                    (0, 3, 0, 1.0, 1.0, 0.0, 1),
                    (1, 3, 1, 1.0, 2.0, 0.0, 0),
                    (2, 3, 2, 1.0, 3.0, 0.0, 1),
                    (0, 4, 0, 1.0, 1.0, 0.0, 0),
                    (1, 4, 1, 1.0, 2.0, 0.0, 1),
                    (2, 4, 2, 1.0, 3.0, 0.0, 0),
                    (0, 5, 0, 1.0, 1.0, 0.0, 1),
                    (1, 5, 1, 1.0, 2.0, 0.0, 0),
                    (2, 5, 2, 1.0, 3.0, 0.0, 1),
                    (0, 6, 0, 1.0, 1.0, 0.0, 0),
                    (1, 6, 1, 2.0, 2.0, 0.0, 1),
                    (2, 6, 2, 2.0, 3.0, 0.0, 0),
                    (0, 7, 0, 1.0, 1.0, 0.0, 1),
                    (1, 7, 1, 2.0, 2.0, 0.0, 0),
                    (2, 7, 2, 2.0, 3.0, 0.0, 1),
                    (0, 8, 0, 1.0, 1.0, 0.0, 0),
                    (1, 8, 1, 1.0, 2.0, 0.0, 1),
                    (2, 8, 2, 1.0, 3.0, 0.0, 0),
                    (0, 9, 0, 1.0, 1.0, 0.0, 1),
                    (1, 9, 1, 1.0, 2.0, 0.0, 0),
                    (2, 9, 2, 1.0, 3.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_init_cash(self):
        assert_records_close(
            from_signals_both(close=price_wide, size=1.0, init_cash=[0.0, 1.0, 100.0]).order_records,
            np.array(
                [
                    (0, 0, 3, 1.0, 4.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 2.0, 4.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 3, 2.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(close=price_wide, size=1.0, init_cash=[0.0, 1.0, 100.0]).order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 3, 1.0, 4.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 3, 1.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(close=price_wide, size=1.0, init_cash=[0.0, 1.0, 100.0]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 3, 0.25, 4.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 3, 0.5, 4.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.0, 1),
                    (1, 2, 3, 1.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_signals_both(init_cash=np.inf).order_records
        with pytest.raises(Exception):
            from_signals_longonly(init_cash=np.inf).order_records
        with pytest.raises(Exception):
            from_signals_shortonly(init_cash=np.inf).order_records

    def test_init_position(self):
        pf = vbt.Portfolio.from_signals(
            close=1,
            entries=False,
            exits=True,
            init_cash=0.0,
            init_position=1.0,
            direction="longonly",
        )
        assert pf.init_position == 1.0
        assert_records_close(pf.order_records, np.array([(0, 0, 0, 1.0, 1.0, 0.0, 1)], dtype=order_dt))

    def test_group_by(self):
        pf = from_signals_both(close=price_wide, group_by=np.array([0, 0, 1]))
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 200.0, 4.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 3, 200.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_index_equal(pf.wrapper.grouper.group_by, pd.Index([0, 0, 1], dtype="int64"))
        assert_series_equal(
            pf.init_cash,
            pd.Series([200.0, 100.0], index=pd.Index([0, 1], dtype="int64")).rename("init_cash"),
        )
        assert not pf.cash_sharing

    def test_cash_sharing(self):
        pf = from_signals_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 200.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_index_equal(pf.wrapper.grouper.group_by, pd.Index([0, 0, 1], dtype="int64"))
        assert_series_equal(
            pf.init_cash,
            pd.Series([100.0, 100.0], index=pd.Index([0, 1], dtype="int64")).rename("init_cash"),
        )
        assert pf.cash_sharing
        with pytest.raises(Exception):
            pf.regroup(group_by=False)

    def test_call_seq(self):
        pf = from_signals_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 200.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        )
        pf = from_signals_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True, call_seq="reversed")
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 3, 200.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        pf = from_signals_both(
            close=price_wide,
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            call_seq="random",
            seed=seed,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 3, 200.0, 4.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 200.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        kwargs = dict(
            close=1.0,
            entries=pd.DataFrame(
                [
                    [False, False, True],
                    [False, True, False],
                    [True, False, False],
                    [False, False, True],
                    [False, True, False],
                ]
            ),
            exits=pd.DataFrame(
                [
                    [False, False, False],
                    [False, False, True],
                    [False, True, False],
                    [True, False, False],
                    [False, False, True],
                ]
            ),
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq="auto",
        )
        pf = from_signals_both(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 2, 200.0, 1.0, 0.0, 0),
                    (1, 0, 3, 200.0, 1.0, 0.0, 1),
                    (0, 1, 1, 200.0, 1.0, 0.0, 0),
                    (1, 1, 2, 200.0, 1.0, 0.0, 1),
                    (2, 1, 4, 200.0, 1.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 1.0, 0.0, 1),
                    (2, 2, 3, 200.0, 1.0, 0.0, 0),
                    (3, 2, 4, 200.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 0, 1]]),
        )
        pf = from_signals_longonly(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 2, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 100.0, 1.0, 0.0, 1),
                    (0, 1, 1, 100.0, 1.0, 0.0, 0),
                    (1, 1, 2, 100.0, 1.0, 0.0, 1),
                    (2, 1, 4, 100.0, 1.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 100.0, 1.0, 0.0, 1),
                    (2, 2, 3, 100.0, 1.0, 0.0, 0),
                    (3, 2, 4, 100.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 0, 1]]),
        )
        pf = from_signals_shortonly(**kwargs)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 2, 100.0, 1.0, 0.0, 1),
                    (1, 0, 3, 100.0, 1.0, 0.0, 0),
                    (0, 1, 4, 100.0, 1.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 1, 100.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[2, 0, 1], [1, 0, 2], [0, 1, 2], [2, 1, 0], [1, 0, 2]]),
        )
        pf = from_signals_longonly(**kwargs, size=1.0, size_type="percent")
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 2, 100.0, 1.0, 0.0, 0),
                    (1, 0, 3, 100.0, 1.0, 0.0, 1),
                    (0, 1, 1, 100.0, 1.0, 0.0, 0),
                    (1, 1, 2, 100.0, 1.0, 0.0, 1),
                    (2, 1, 4, 100.0, 1.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 100.0, 1.0, 0.0, 1),
                    (2, 2, 3, 100.0, 1.0, 0.0, 0),
                    (3, 2, 4, 100.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 0, 1], [1, 0, 2], [0, 1, 2], [2, 0, 1]]),
        )
        _ = from_signals_both(attach_call_seq=False, **kwargs)
        _ = from_signals_longonly(attach_call_seq=False, **kwargs)
        _ = from_signals_shortonly(attach_call_seq=False, **kwargs)

    def test_sl_stop(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 0),
                    (0, 1, 0, 20.0, 5.0, 0.0, 0),
                    (1, 1, 1, 20.0, 4.5, 0.0, 1),
                    (0, 2, 0, 20.0, 5.0, 0.0, 0),
                    (1, 2, 3, 20.0, 2.5, 0.0, 1),
                    (0, 3, 0, 20.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 1),
                    (0, 1, 0, 20.0, 5.0, 0.0, 1),
                    (0, 2, 0, 20.0, 5.0, 0.0, 1),
                    (0, 3, 0, 20.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 0),
                    (0, 1, 0, 20.0, 5.0, 0.0, 0),
                    (1, 1, 1, 20.0, 4.25, 0.0, 1),
                    (0, 2, 0, 20.0, 5.0, 0.0, 0),
                    (1, 2, 1, 20.0, 4.25, 0.0, 1),
                    (0, 3, 0, 20.0, 5.0, 0.0, 0),
                    (1, 3, 1, 20.0, 4.0, 0.0, 1),
                    (0, 4, 0, 20.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 1),
                    (0, 1, 0, 20.0, 5.0, 0.0, 1),
                    (0, 2, 0, 20.0, 5.0, 0.0, 1),
                    (0, 3, 0, 20.0, 5.0, 0.0, 1),
                    (0, 4, 0, 20.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (0, 3, 0, 100.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (1, 1, 1, 100.0, 1.5, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 3, 50.0, 4.0, 0.0, 0),
                    (0, 3, 0, 100.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (0, 3, 0, 100.0, 1.0, 0.0, 0),
                    (0, 4, 0, 100.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (1, 1, 1, 100.0, 1.75, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 1, 100.0, 1.75, 0.0, 0),
                    (0, 3, 0, 100.0, 1.0, 0.0, 1),
                    (1, 3, 1, 100.0, 2.0, 0.0, 0),
                    (0, 4, 0, 100.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_tsl_stop(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([4.0, 5.0, 4.0, 3.0, 2.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 25.0, 4.0, 0.0, 0),
                    (0, 1, 0, 25.0, 4.0, 0.0, 0),
                    (1, 1, 2, 25.0, 4.5, 0.0, 1),
                    (0, 2, 0, 25.0, 4.0, 0.0, 0),
                    (1, 2, 4, 25.0, 2.5, 0.0, 1),
                    (0, 3, 0, 25.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 25.0, 4.0, 0.0, 1),
                    (0, 1, 0, 25.0, 4.0, 0.0, 1),
                    (1, 1, 1, 25.0, 4.4, 0.0, 0),
                    (0, 2, 0, 25.0, 4.0, 0.0, 1),
                    (0, 3, 0, 25.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.15, 0.2, 0.25, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 25.0, 4.0, 0.0, 0),
                    (0, 1, 0, 25.0, 4.0, 0.0, 0),
                    (1, 1, 2, 25.0, 4.25, 0.0, 1),
                    (0, 2, 0, 25.0, 4.0, 0.0, 0),
                    (1, 2, 2, 25.0, 4.25, 0.0, 1),
                    (0, 3, 0, 25.0, 4.0, 0.0, 0),
                    (1, 3, 2, 25.0, 4.125, 0.0, 1),
                    (0, 4, 0, 25.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.15, 0.2, 0.25, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 25.0, 4.0, 0.0, 1),
                    (0, 1, 0, 25.0, 4.0, 0.0, 1),
                    (1, 1, 1, 25.0, 5.25, 0.0, 0),
                    (0, 2, 0, 25.0, 4.0, 0.0, 1),
                    (1, 2, 1, 25.0, 5.25, 0.0, 0),
                    (0, 3, 0, 25.0, 4.0, 0.0, 1),
                    (1, 3, 1, 25.0, 5.25, 0.0, 0),
                    (0, 4, 0, 25.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

        close = pd.Series([2.0, 1.0, 2.0, 3.0, 4.0], index=price.index)
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 2.0, 0.0, 0),
                    (0, 1, 0, 50.0, 2.0, 0.0, 0),
                    (1, 1, 1, 50.0, 1.0, 0.0, 1),
                    (0, 2, 0, 50.0, 2.0, 0.0, 0),
                    (0, 3, 0, 50.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 2.0, 0.0, 1),
                    (0, 1, 0, 50.0, 2.0, 0.0, 1),
                    (1, 1, 2, 50.0, 1.5, 0.0, 0),
                    (0, 2, 0, 50.0, 2.0, 0.0, 1),
                    (1, 2, 4, 50.0, 4.0, 0.0, 0),
                    (0, 3, 0, 50.0, 2.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tsl_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 2.0, 0.0, 0),
                    (0, 1, 0, 50.0, 2.0, 0.0, 0),
                    (1, 1, 1, 50.0, 0.75, 0.0, 1),
                    (0, 2, 0, 50.0, 2.0, 0.0, 0),
                    (1, 2, 1, 50.0, 0.5, 0.0, 1),
                    (0, 3, 0, 50.0, 2.0, 0.0, 0),
                    (0, 4, 0, 50.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tsl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 2.0, 0.0, 1),
                    (0, 1, 0, 50.0, 2.0, 0.0, 1),
                    (1, 1, 2, 50.0, 1.75, 0.0, 0),
                    (0, 2, 0, 50.0, 2.0, 0.0, 1),
                    (1, 2, 2, 50.0, 1.75, 0.0, 0),
                    (0, 3, 0, 50.0, 2.0, 0.0, 1),
                    (1, 3, 2, 50.0, 1.75, 0.0, 0),
                    (0, 4, 0, 50.0, 2.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_ttp_stop(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([4.0, 3.0, 5.0, 4.0, 2.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 25.0, 4.0, 0.0, 0),
                    (0, 1, 0, 25.0, 4.0, 0.0, 0),
                    (1, 1, 3, 25.0, 4.5, 0.0, 1),
                    (0, 2, 0, 25.0, 4.0, 0.0, 0),
                    (0, 3, 0, 25.0, 4.0, 0.0, 0),
                    (1, 3, 4, 25.0, 2.5, 0.0, 1),
                    (0, 4, 0, 25.0, 4.0, 0.0, 0),
                    (0, 5, 0, 25.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 25.0, 4.0, 0.0, 1),
                    (0, 1, 0, 25.0, 4.0, 0.0, 1),
                    (1, 1, 2, 25.0, 3.3000000000000003, 0.0, 0),
                    (0, 2, 0, 25.0, 4.0, 0.0, 1),
                    (0, 3, 0, 25.0, 4.0, 0.0, 1),
                    (1, 3, 2, 25.0, 4.5, 0.0, 0),
                    (0, 4, 0, 25.0, 4.0, 0.0, 1),
                    (0, 5, 0, 25.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 25.0, 4.0, 0.0, 0),
                    (0, 1, 0, 25.0, 4.0, 0.0, 0),
                    (1, 1, 3, 25.0, 4.25, 0.0, 1),
                    (0, 2, 0, 25.0, 4.0, 0.0, 0),
                    (0, 3, 0, 25.0, 4.0, 0.0, 0),
                    (1, 3, 4, 25.0, 2.25, 0.0, 1),
                    (0, 4, 0, 25.0, 4.0, 0.0, 0),
                    (0, 5, 0, 25.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 25.0, 4.0, 0.0, 1),
                    (0, 1, 0, 25.0, 4.0, 0.0, 1),
                    (1, 1, 2, 25.0, 5.25, 0.0, 0),
                    (0, 2, 0, 25.0, 4.0, 0.0, 1),
                    (0, 3, 0, 25.0, 4.0, 0.0, 1),
                    (1, 3, 2, 25.0, 5.25, 0.0, 0),
                    (0, 4, 0, 25.0, 4.0, 0.0, 1),
                    (0, 5, 0, 25.0, 4.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

        close = pd.Series([3.0, 4.0, 2.0, 3.0, 4.0], index=price.index)
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (0, 1, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (1, 1, 2, 33.333333333333336, 3.6, 0.0, 1),
                    (0, 2, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (0, 3, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (1, 3, 2, 33.333333333333336, 2.0, 0.0, 1),
                    (0, 4, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (0, 5, 0, 33.333333333333336, 3.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (0, 1, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (1, 1, 3, 33.333333333333336, 2.2, 0.0, 0),
                    (0, 2, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (0, 3, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (1, 3, 3, 33.333333333333336, 3.0, 0.0, 0),
                    (0, 4, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (0, 5, 0, 33.333333333333336, 3.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (0, 1, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (1, 1, 2, 33.333333333333336, 1.75, 0.0, 1),
                    (0, 2, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (1, 2, 2, 33.333333333333336, 1.75, 0.0, 1),
                    (0, 3, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (1, 3, 2, 33.333333333333336, 1.75, 0.0, 1),
                    (0, 4, 0, 33.333333333333336, 3.0, 0.0, 0),
                    (1, 4, 2, 33.333333333333336, 1.75, 0.0, 1),
                    (0, 5, 0, 33.333333333333336, 3.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                ttp_th=[[np.nan, 0.1, 0.5, 0.1, 0.5, np.inf]],
                ttp_stop=[[np.nan, 0.1, 0.1, 0.5, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (0, 1, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (1, 1, 3, 33.333333333333336, 2.75, 0.0, 0),
                    (0, 2, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (1, 2, 3, 33.333333333333336, 2.75, 0.0, 0),
                    (0, 3, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (1, 3, 3, 33.333333333333336, 2.75, 0.0, 0),
                    (0, 4, 0, 33.333333333333336, 3.0, 0.0, 1),
                    (1, 4, 3, 33.333333333333336, 2.75, 0.0, 0),
                    (0, 5, 0, 33.333333333333336, 3.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_tp_stop(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 0),
                    (0, 1, 0, 20.0, 5.0, 0.0, 0),
                    (0, 2, 0, 20.0, 5.0, 0.0, 0),
                    (0, 3, 0, 20.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 1),
                    (0, 1, 0, 20.0, 5.0, 0.0, 1),
                    (1, 1, 1, 20.0, 4.5, 0.0, 0),
                    (0, 2, 0, 20.0, 5.0, 0.0, 1),
                    (1, 2, 3, 20.0, 2.5, 0.0, 0),
                    (0, 3, 0, 20.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.1, 0.5, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 0),
                    (0, 1, 0, 20.0, 5.0, 0.0, 0),
                    (0, 2, 0, 20.0, 5.0, 0.0, 0),
                    (0, 3, 0, 20.0, 5.0, 0.0, 0),
                    (0, 4, 0, 20.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.1, 0.15, 0.2, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 1),
                    (0, 1, 0, 20.0, 5.0, 0.0, 1),
                    (1, 1, 1, 20.0, 4.25, 0.0, 0),
                    (0, 2, 0, 20.0, 5.0, 0.0, 1),
                    (1, 2, 1, 20.0, 4.25, 0.0, 0),
                    (0, 3, 0, 20.0, 5.0, 0.0, 1),
                    (1, 3, 1, 20.0, 4.0, 0.0, 0),
                    (0, 4, 0, 20.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)
        open = close - 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 1.5, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 3, 100.0, 4.0, 0.0, 1),
                    (0, 3, 0, 100.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (0, 3, 0, 100.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=[[np.nan, 0.5, 3.0, np.inf]],
            ).order_records,
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 1.75, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 100.0, 1.75, 0.0, 1),
                    (0, 3, 0, 100.0, 1.0, 0.0, 0),
                    (1, 3, 1, 100.0, 2.0, 0.0, 1),
                    (0, 4, 0, 100.0, 1.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                tp_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (0, 3, 0, 100.0, 1.0, 0.0, 1),
                    (0, 4, 0, 100.0, 1.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_stop_entry_price(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                stop_entry_price="val_price",
                stop_exit_price="stoplimit",
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 0, 1, 16.52892561983471, 4.25, 0.0, 1),
                    (0, 1, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 1, 2, 16.52892561983471, 2.625, 0.0, 1),
                    (0, 2, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 2, 4, 16.52892561983471, 1.25, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                stop_entry_price="price",
                stop_exit_price="stoplimit",
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 0, 1, 16.52892561983471, 4.25, 0.0, 1),
                    (0, 1, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 1, 2, 16.52892561983471, 2.75, 0.0, 1),
                    (0, 2, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 2, 4, 16.52892561983471, 1.25, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                stop_entry_price="fillprice",
                stop_exit_price="stoplimit",
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 0, 1, 16.52892561983471, 4.25, 0.0, 1),
                    (0, 1, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 1, 2, 16.52892561983471, 3.0250000000000004, 0.0, 1),
                    (0, 2, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 2, 3, 16.52892561983471, 1.5125000000000002, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                val_price=1.05 * close,
                stop_entry_price="close",
                stop_exit_price="stoplimit",
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 0, 1, 16.52892561983471, 4.25, 0.0, 1),
                    (0, 1, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 1, 2, 16.52892561983471, 2.5, 0.0, 1),
                    (0, 2, 0, 16.52892561983471, 6.050000000000001, 0.0, 0),
                    (1, 2, 4, 16.52892561983471, 1.25, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_stop_exit_price(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                stop_exit_price="stoplimit",
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 0, 1, 16.528926, 4.25, 0.0, 1),
                    (0, 1, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 1, 2, 16.528926, 2.5, 0.0, 1),
                    (0, 2, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 2, 4, 16.528926, 1.25, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                stop_exit_price="stopmarket",
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 0, 1, 16.528926, 3.825, 0.0, 1),
                    (0, 1, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 1, 2, 16.528926, 2.25, 0.0, 1),
                    (0, 2, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 2, 4, 16.528926, 1.125, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                stop_exit_price="close",
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 0, 1, 16.528926, 3.6, 0.0, 1),
                    (0, 1, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 1, 2, 16.528926, 2.7, 0.0, 1),
                    (0, 2, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 2, 4, 16.528926, 0.9, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                open=open,
                high=high,
                low=low,
                sl_stop=[[0.05, 0.5, 0.75]],
                price=1.1 * close,
                stop_exit_price="price",
                slippage=0.1,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 0, 1, 16.528926, 3.9600000000000004, 0.0, 1),
                    (0, 1, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 1, 2, 16.528926, 2.97, 0.0, 1),
                    (0, 2, 0, 16.528926, 6.05, 0.0, 0),
                    (1, 2, 4, 16.528926, 0.9900000000000001, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_upon_stop_exit(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                size=1,
                sl_stop=0.1,
                upon_stop_exit=[["close", "closereduce", "reverse", "reversereduce"]],
                accumulate=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 5.0, 0.0, 0),
                    (1, 0, 1, 1.0, 4.5, 0.0, 1),
                    (0, 1, 0, 1.0, 5.0, 0.0, 0),
                    (1, 1, 1, 1.0, 4.5, 0.0, 1),
                    (0, 2, 0, 1.0, 5.0, 0.0, 0),
                    (1, 2, 1, 2.0, 4.5, 0.0, 1),
                    (0, 3, 0, 1.0, 5.0, 0.0, 0),
                    (1, 3, 1, 1.0, 4.5, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                size=1,
                sl_stop=0.1,
                upon_stop_exit=[["close", "closereduce", "reverse", "reversereduce"]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 5.0, 0.0, 0),
                    (1, 0, 1, 1.0, 4.5, 0.0, 1),
                    (0, 1, 0, 1.0, 5.0, 0.0, 0),
                    (1, 1, 1, 1.0, 4.5, 0.0, 1),
                    (0, 2, 0, 1.0, 5.0, 0.0, 0),
                    (1, 2, 1, 2.0, 4.5, 0.0, 1),
                    (0, 3, 0, 1.0, 5.0, 0.0, 0),
                    (1, 3, 1, 2.0, 4.5, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_upon_stop_update(self):
        entries = pd.Series([True, True, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        sl_stop = pd.Series([0.4, np.nan, np.nan, np.nan, np.nan])
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                accumulate=True,
                size=1.0,
                sl_stop=sl_stop,
                upon_stop_update=[["keep", "override", "overridenan"]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 5.0, 0.0, 0),
                    (1, 0, 1, 1.0, 4.0, 0.0, 0),
                    (2, 0, 2, 2.0, 3.0, 0.0, 1),
                    (0, 1, 0, 1.0, 5.0, 0.0, 0),
                    (1, 1, 1, 1.0, 4.0, 0.0, 0),
                    (2, 1, 2, 2.0, 3.0, 0.0, 1),
                    (0, 2, 0, 1.0, 5.0, 0.0, 0),
                    (1, 2, 1, 1.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        sl_stop = pd.Series([0.4, 0.4, np.nan, np.nan, np.nan])
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                accumulate=True,
                size=1.0,
                sl_stop=sl_stop,
                upon_stop_update=[["keep", "override"]],
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 5.0, 0.0, 0),
                    (1, 0, 1, 1.0, 4.0, 0.0, 0),
                    (2, 0, 2, 2.0, 3.0, 0.0, 1),
                    (0, 1, 0, 1.0, 5.0, 0.0, 0),
                    (1, 1, 1, 1.0, 4.0, 0.0, 0),
                    (2, 1, 3, 2.0, 2.4, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_signal_priority(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, True, False], index=price.index)

        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[0.1, 0.5]],
                signal_priority="stop",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 0),
                    (1, 0, 1, 20.0, 4.5, 0.0, 1),
                    (2, 0, 3, 45.0, 2.0, 0.0, 1),
                    (0, 1, 0, 20.0, 5.0, 0.0, 0),
                    (1, 1, 3, 20.0, 2.5, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[0.1, 0.5]],
                signal_priority="user",
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 20.0, 5.0, 0.0, 0),
                    (1, 0, 1, 20.0, 4.5, 0.0, 1),
                    (2, 0, 3, 45.0, 2.0, 0.0, 1),
                    (0, 1, 0, 20.0, 5.0, 0.0, 0),
                    (1, 1, 3, 40.0, 2.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_adjust_func(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)

        @njit
        def adjust_sl_func_nb(c, dur):
            if c.i - c.sl_init_i >= dur:
                c.sl_curr_stop[c.col] = 0.0

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=np.inf,
                adjust_func_nb=adjust_sl_func_nb,
                adjust_args=(2,),
            ).order_records,
            np.array([(0, 0, 0, 20.0, 5.0, 0.0, 0), (1, 0, 2, 20.0, 5.0, 0.0, 1)], dtype=order_dt),
        )

        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)

        @njit
        def adjust_tp_func_nb(c, dur):
            if c.i - c.tp_init_i >= dur:
                c.tp_curr_stop[c.col] = 0.0

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=np.inf,
                adjust_func_nb=adjust_tp_func_nb,
                adjust_args=(2,),
            ).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 2, 100.0, 1.0, 0.0, 1)], dtype=order_dt),
        )

    def test_max_orders(self):
        assert from_signals_both(close=price_wide).order_records.shape[0] == 6
        assert from_signals_both(close=price_wide, max_orders=2).order_records.shape[0] == 6
        assert from_signals_both(close=price_wide, max_orders=0).order_records.shape[0] == 0
        with pytest.raises(Exception):
            from_signals_both(close=price_wide, max_orders=1)

    def test_max_logs(self):
        assert from_signals_both(close=price_wide, log=True).log_records.shape[0] == 6
        assert from_signals_both(close=price_wide, log=True, max_logs=2).log_records.shape[0] == 6
        assert from_signals_both(close=price_wide, log=True, max_logs=0).log_records.shape[0] == 0
        with pytest.raises(Exception):
            from_signals_both(close=price_wide, log=True, max_logs=1)

    def test_jitted_parallel(self):
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        entries2 = pd.concat((entries, entries.vbt.signals.fshift(1), entries.vbt.signals.fshift(2)), axis=1)
        entries2.columns = price_wide2.columns
        exits2 = pd.concat((exits, exits.vbt.signals.fshift(1), exits.vbt.signals.fshift(2)), axis=1)
        exits2.columns = price_wide2.columns
        pf = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200, 300],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            log=True,
            jitted=dict(parallel=True),
        )
        pf2 = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200, 300],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            log=True,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            jitted=dict(parallel=True),
        )
        pf2 = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)

    def test_chunked(self):
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        entries2 = pd.concat((entries, entries.vbt.signals.fshift(1), entries.vbt.signals.fshift(2)), axis=1)
        entries2.columns = price_wide2.columns
        exits2 = pd.concat((exits, exits.vbt.signals.fshift(1), exits.vbt.signals.fshift(2)), axis=1)
        exits2.columns = price_wide2.columns
        pf = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200, 300],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            log=True,
            chunked=True,
        )
        pf2 = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200, 300],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            log=True,
            chunked=False,
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            chunked=True,
        )
        pf2 = from_signals_both(
            close=price_wide2,
            entries=entries2,
            exits=exits2,
            init_cash=[100, 200],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            chunked=False,
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)

    def test_cash_earnings(self):
        pf = vbt.Portfolio.from_signals(1, cash_earnings=[0, 1, 2, 3], accumulate=True)
        assert_series_equal(pf.cash_earnings, pd.Series([0.0, 1.0, 2.0, 3.0]))
        assert_records_close(
            pf.order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 2, 1.0, 1.0, 0.0, 0), (2, 0, 3, 2.0, 1.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )

    def test_cash_dividends(self):
        pf = vbt.Portfolio.from_signals(1, size=np.inf, cash_dividends=[0, 1, 2, 3], accumulate=True)
        assert_series_equal(pf.cash_earnings, pd.Series([0.0, 100.0, 400.0, 1800.0]))
        assert_records_close(
            pf.order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 2, 100.0, 1.0, 0.0, 0), (2, 0, 3, 400.0, 1.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )

    @pytest.mark.parametrize("test_group_by", [False, np.array([0, 0, 1])])
    @pytest.mark.parametrize("test_cash_sharing", [False, True])
    def test_fill_returns(self, test_group_by, test_cash_sharing):
        assert_frame_equal(
            from_signals_both(
                close=price_wide,
                fill_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_signals_both(
                close=price_wide,
                fill_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )
        assert_frame_equal(
            from_signals_longonly(
                close=price_wide,
                fill_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_signals_longonly(
                close=price_wide,
                fill_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )
        assert_frame_equal(
            from_signals_shortonly(
                close=price_wide,
                fill_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_signals_shortonly(
                close=price_wide,
                fill_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )


# ############# from_holding ############# #


class TestFromHolding:
    def test_from_holding(self):
        assert_records_close(
            vbt.Portfolio.from_holding(price).order_records,
            vbt.Portfolio.from_signals(price, True, False, accumulate=False).order_records,
        )
        entries = pd.Series.vbt.signals.empty_like(price)
        entries.iloc[0] = True
        exits = pd.Series.vbt.signals.empty_like(price)
        exits.iloc[-1] = True
        assert_records_close(
            vbt.Portfolio.from_holding(price, close_at_end=True).order_records,
            vbt.Portfolio.from_signals(price, entries, exits, accumulate=False).order_records,
        )


# ############# from_random_signals ############# #


class TestFromRandomSignals:
    def test_from_random_n(self):
        result = vbt.Portfolio.from_random_signals(price, n=2, seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [False, True, False, True, False],
                [False, False, True, False, True],
            ).order_records,
        )
        assert_index_equal(result.wrapper.index, price.vbt.wrapper.index)
        assert_index_equal(result.wrapper.columns, price.vbt.wrapper.columns)
        result = vbt.Portfolio.from_random_signals(price, n=[1, 2], seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [[False, False], [False, True], [False, False], [True, True], [False, False]],
                [[False, False], [False, False], [False, True], [False, False], [True, True]],
            ).order_records,
        )
        assert_index_equal(
            result.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        assert_index_equal(result.wrapper.columns, pd.Index([1, 2], dtype="int64", name="randnx_n"))

    def test_from_random_prob(self):
        result = vbt.Portfolio.from_random_signals(price, prob=0.5, seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [True, False, False, False, False],
                [False, False, False, False, True],
            ).order_records,
        )
        assert_index_equal(result.wrapper.index, price.vbt.wrapper.index)
        assert_index_equal(result.wrapper.columns, price.vbt.wrapper.columns)
        result = vbt.Portfolio.from_random_signals(price, prob=[0.25, 0.5], seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [[False, True], [False, False], [False, False], [False, False], [True, False]],
                [[False, False], [False, True], [False, False], [False, False], [False, False]],
            ).order_records,
        )
        assert_index_equal(
            result.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        assert_index_equal(
            result.wrapper.columns,
            pd.MultiIndex.from_tuples([(0.25, 0.25), (0.5, 0.5)], names=["rprobnx_entry_prob", "rprobnx_exit_prob"]),
        )
