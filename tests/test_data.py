import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import pytz

import vectorbtpro as vbt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import to_timezone

seed = 42


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True


def teardown_module():
    vbt.settings.reset()


# ############# base.py ############# #


class MyData(vbt.Data):
    @classmethod
    def fetch_symbol(
        cls,
        symbol,
        shape=(5, 3),
        start_date=datetime(2020, 1, 1),
        columns=None,
        index_mask=None,
        column_mask=None,
        return_arr=False,
        tz_localize=None,
        is_update=False,
        return_none=False,
        return_empty=False,
        raise_error=False,
    ):
        if raise_error:
            raise ValueError()
        if return_none:
            return None
        if return_empty:
            if len(shape) == 2:
                a = np.empty((0, shape[1]), dtype=object)
            else:
                a = np.empty((0,), dtype=object)
            if return_arr:
                return a
            if len(shape) == 2:
                return pd.DataFrame(a, columns=columns)
            return pd.Series(a, name=columns)
        np.random.seed(seed)
        a = np.empty(shape, dtype=object)
        if a.ndim == 1:
            for i in range(a.shape[0]):
                a[i] = str(symbol) + "_" + str(i)
                if is_update:
                    a[i] += "_u"
        else:
            for col in range(a.shape[1]):
                for i in range(a.shape[0]):
                    a[i, col] = str(symbol) + "_" + str(col) + "_" + str(i)
                    if is_update:
                        a[i, col] += "_u"
        if return_arr:
            return a
        index = [start_date + timedelta(days=i) for i in range(a.shape[0])]
        if a.ndim == 1:
            sr = pd.Series(a, index=index, name=columns)
            if index_mask is not None:
                sr = sr.loc[index_mask]
            if tz_localize is not None:
                sr = sr.tz_localize(tz_localize)
            return sr
        df = pd.DataFrame(a, index=index, columns=columns)
        if index_mask is not None:
            df = df.loc[index_mask]
        if column_mask is not None:
            df = df.loc[:, column_mask]
        if tz_localize is not None:
            df = df.tz_localize(tz_localize)
        return df

    def update_symbol(self, symbol, n=1, **kwargs):
        fetch_kwargs = self.select_symbol_kwargs(symbol, self.fetch_kwargs)
        start_date = self.last_index[symbol]
        shape = fetch_kwargs.pop("shape", (5, 3))
        new_shape = (n, shape[1]) if len(shape) > 1 else (n,)
        kwargs = merge_dicts(fetch_kwargs, dict(start_date=start_date), kwargs)
        return self.fetch_symbol(symbol, shape=new_shape, is_update=True, **kwargs)


class TestData:
    def test_config(self, tmp_path):
        data = MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert MyData.loads(data.dumps()) == data
        data.save(tmp_path / "data")
        assert MyData.load(tmp_path / "data") == data

    def test_fetch(self):
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), return_arr=True).data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_4"]),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(0, shape=(5, 3), return_arr=True).data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                ]
            ),
        )
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,)).data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_4"], index=index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), columns="feat0").data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_4"], index=index, name="feat0"),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(0, shape=(5, 3)).data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                ],
                index=index,
                columns=pd.Index(["feat0", "feat1", "feat2"], dtype="object"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,)).data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_4"], index=index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,)).data[1],
            pd.Series(["1_0", "1_1", "1_2", "1_3", "1_4"], index=index),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3)).data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3)).data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", "1_2_0"],
                    ["1_0_1", "1_1_1", "1_2_1"],
                    ["1_0_2", "1_1_2", "1_2_2"],
                    ["1_0_3", "1_1_3", "1_2_3"],
                    ["1_0_4", "1_1_4", "1_2_4"],
                ],
                index=index,
            ),
        )
        index2 = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=pytz.utc,
        ).tz_convert(to_timezone("Europe/Berlin"))
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), tz_localize="UTC", tz_convert="Europe/Berlin").data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_4"], index=index2),
        )
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan").data[0],
            pd.Series([np.nan, "0_1", "0_2", "0_3", "0_4"], index=index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan").data[1],
            pd.Series(["1_0", "1_1", "1_2", "1_3", np.nan], index=index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop").data[0],
            pd.Series(["0_1", "0_2", "0_3"], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop").data[1],
            pd.Series(["1_1", "1_2", "1_3"], index=index[1:4]),
        )
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            ).data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, "0_1_1", "0_2_1"],
                    [np.nan, "0_1_2", "0_2_2"],
                    [np.nan, "0_1_3", "0_2_3"],
                    [np.nan, "0_1_4", "0_2_4"],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            ).data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", np.nan],
                    ["1_0_1", "1_1_1", np.nan],
                    ["1_0_2", "1_1_2", np.nan],
                    ["1_0_3", "1_1_3", np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).data[0],
            pd.DataFrame(
                [["0_1_1"], ["0_1_2"], ["0_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            ).data[1],
            pd.DataFrame(
                [["1_1_1"], ["1_1_2"], ["1_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert len(MyData.fetch([0, 1], shape=(5, 3), return_none=vbt.symbol_dict({0: True, 1: False})).symbols) == 1
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), return_none=vbt.symbol_dict({0: True, 1: False})).data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", "1_2_0"],
                    ["1_0_1", "1_1_1", "1_2_1"],
                    ["1_0_2", "1_1_2", "1_2_2"],
                    ["1_0_3", "1_1_3", "1_2_3"],
                    ["1_0_4", "1_1_4", "1_2_4"],
                ],
                index=index,
            ),
        )
        assert len(MyData.fetch([0, 1], shape=(5, 3), return_empty=vbt.symbol_dict({0: True, 1: False})).symbols) == 1
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), return_empty=vbt.symbol_dict({0: True, 1: False})).data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", "1_2_0"],
                    ["1_0_1", "1_1_1", "1_2_1"],
                    ["1_0_2", "1_1_2", "1_2_2"],
                    ["1_0_3", "1_1_3", "1_2_3"],
                    ["1_0_4", "1_1_4", "1_2_4"],
                ],
                index=index,
            ),
        )
        assert (
            len(
                MyData.fetch(
                    [0, 1], shape=(5, 3), raise_error=vbt.symbol_dict({0: True, 1: False}), skip_on_error=True
                ).symbols
            )
            == 1
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1], shape=(5, 3), raise_error=vbt.symbol_dict({0: True, 1: False}), skip_on_error=True
            ).data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", "1_2_0"],
                    ["1_0_1", "1_1_1", "1_2_1"],
                    ["1_0_2", "1_1_2", "1_2_2"],
                    ["1_0_3", "1_1_3", "1_2_3"],
                    ["1_0_4", "1_1_4", "1_2_4"],
                ],
                index=index,
            ),
        )
        with pytest.raises(Exception):
            MyData.fetch([0, 1], shape=(5, 3), raise_error=vbt.symbol_dict({0: True, 1: False}), skip_on_error=False)
        with pytest.raises(Exception):
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="raise",
                missing_columns="nan",
            )
        with pytest.raises(Exception):
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="raise",
            )
        with pytest.raises(Exception):
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="test",
                missing_columns="nan",
            )
        with pytest.raises(Exception):
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="test",
            )
        with pytest.raises(Exception):
            MyData.fetch([0, 1], shape=(5, 3), return_none=True)
        with pytest.raises(Exception):
            MyData.fetch([0, 1], shape=(5, 3), return_empty=True)
        with pytest.raises(Exception):
            MyData.fetch([0, 1], shape=(5, 3), raise_error=True, skip_on_error=False)
        with pytest.raises(Exception):
            MyData.fetch([0, 1], shape=(5, 3), raise_error=True, skip_on_error=True)

    def test_update(self):
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), return_arr=True).update().data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_0_u"]),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), return_arr=True).update(concat=False).data[0],
            pd.Series(["0_0_u"], index=pd.Index([4], dtype="int64")),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), return_arr=True).update(n=2).data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_0_u", "0_1_u"]),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), return_arr=True).update(n=2, concat=False).data[0],
            pd.Series(["0_0_u", "0_1_u"], index=pd.Index([4, 5], dtype="int64")),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(0, shape=(5, 3), return_arr=True).update().data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_0_u", "0_1_0_u", "0_2_0_u"],
                ]
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(0, shape=(5, 3), return_arr=True).update(concat=False).data[0],
            pd.DataFrame(
                [
                    ["0_0_0_u", "0_1_0_u", "0_2_0_u"],
                ],
                index=pd.Index([4], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(0, shape=(5, 3), return_arr=True).update(n=2).data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_0_u", "0_1_0_u", "0_2_0_u"],
                    ["0_0_1_u", "0_1_1_u", "0_2_1_u"],
                ]
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(0, shape=(5, 3), return_arr=True).update(n=2, concat=False).data[0],
            pd.DataFrame(
                [
                    ["0_0_0_u", "0_1_0_u", "0_2_0_u"],
                    ["0_0_1_u", "0_1_1_u", "0_2_1_u"],
                ],
                index=pd.Index([4, 5], dtype="int64"),
            ),
        )
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,)).update().data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_0_u"], index=index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,)).update(concat=False).data[0],
            pd.Series(["0_0_u"], index=index[[-1]]),
        )
        updated_index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
                "2020-01-06 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,)).update(n=2).data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_0_u", "0_1_u"], index=updated_index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,)).update(n=2, concat=False).data[0],
            pd.Series(
                ["0_0_u", "0_1_u"],
                index=pd.DatetimeIndex(
                    ["2020-01-05 00:00:00+00:00", "2020-01-06 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
            ),
        )
        index2 = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=pytz.utc,
        ).tz_convert(to_timezone("Europe/Berlin"))
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), tz_localize="UTC", tz_convert="Europe/Berlin").update(tz_localize=None).data[0],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_0_u"], index=index2),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), tz_localize="UTC", tz_convert="Europe/Berlin")
            .update(tz_localize=None, concat=False)
            .data[0],
            pd.Series(["0_0_u"], index=index2[[-1]]),
        )
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        update_index_mask = vbt.symbol_dict({0: [True], 1: [False]})
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(index_mask=update_index_mask)
            .data[0],
            pd.Series([np.nan, "0_1", "0_2", "0_3", "0_0_u"], index=index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(index_mask=update_index_mask)
            .data[1],
            pd.Series(["1_0", "1_1", "1_2", "1_3", np.nan], index=index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(index_mask=update_index_mask, concat=False)
            .data[0],
            pd.Series(["0_0_u"], index=index[[-1]], dtype=object),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(index_mask=update_index_mask, concat=False)
            .data[1],
            pd.Series([np.nan], index=index[[-1]], dtype=object),
        )
        update_index_mask2 = vbt.symbol_dict({0: [True, False, False], 1: [True, False, True]})
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(n=3, index_mask=update_index_mask2)
            .data[0],
            pd.Series([np.nan, "0_1", "0_2", "0_3", "0_0_u", np.nan], index=updated_index),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(n=3, index_mask=update_index_mask2)
            .data[1],
            pd.Series(
                [
                    "1_0",
                    "1_1",
                    "1_2",
                    "1_0_u",
                    np.nan,
                    "1_2_u",
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data[0],
            pd.Series(["0_3", "0_0_u", np.nan], index=updated_index[-3:]),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="nan")
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data[1],
            pd.Series(
                [
                    "1_0_u",
                    np.nan,
                    "1_2_u",
                ],
                index=updated_index[-3:],
            ),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(index_mask=update_index_mask)
            .data[0],
            pd.Series(["0_1", "0_2", "0_3"], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(index_mask=update_index_mask)
            .data[1],
            pd.Series(["1_1", "1_2", "1_3"], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(index_mask=update_index_mask, concat=False)
            .data[0],
            pd.Series([], index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None), dtype=object),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(index_mask=update_index_mask, concat=False)
            .data[1],
            pd.Series([], index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None), dtype=object),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(n=3, index_mask=update_index_mask2)
            .data[0],
            pd.Series(["0_1", "0_2", "0_3"], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(n=3, index_mask=update_index_mask2)
            .data[1],
            pd.Series(["1_1", "1_2", "1_0_u"], index=index[1:4]),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data[0],
            pd.Series(
                ["0_3"], index=pd.DatetimeIndex(["2020-01-04 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None)
            ),
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5,), index_mask=index_mask, missing_index="drop")
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data[1],
            pd.Series(
                ["1_0_u"], index=pd.DatetimeIndex(["2020-01-04 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None)
            ),
        )
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask)
            .data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, "0_1_1", "0_2_1"],
                    [np.nan, "0_1_2", "0_2_2"],
                    [np.nan, "0_1_3", "0_2_3"],
                    [np.nan, "0_1_0_u", "0_2_0_u"],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask)
            .data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", np.nan],
                    ["1_0_1", "1_1_1", np.nan],
                    ["1_0_2", "1_1_2", np.nan],
                    ["1_0_3", "1_1_3", np.nan],
                    [np.nan, np.nan, np.nan],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data[0],
            pd.DataFrame(
                [
                    [np.nan, "0_1_0_u", "0_2_0_u"],
                ],
                index=pd.DatetimeIndex(["2020-01-05 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None),
            ).astype({0: float, 1: object, 2: object}),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data[1],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                ],
                index=pd.DatetimeIndex(["2020-01-05 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None),
            ).astype({0: object, 1: object, 2: float}),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data[0],
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, "0_1_1", "0_2_1"],
                    [np.nan, "0_1_2", "0_2_2"],
                    [np.nan, "0_1_3", "0_2_3"],
                    [np.nan, "0_1_0_u", "0_2_0_u"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", np.nan],
                    ["1_0_1", "1_1_1", np.nan],
                    ["1_0_2", "1_1_2", np.nan],
                    ["1_0_0_u", "1_1_0_u", np.nan],
                    [np.nan, np.nan, np.nan],
                    ["1_0_2_u", "1_1_2_u", np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data[0],
            pd.DataFrame(
                [
                    [np.nan, "0_1_3", "0_2_3"],
                    [np.nan, "0_1_0_u", "0_2_0_u"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index[3:],
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="nan",
                missing_columns="nan",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data[1],
            pd.DataFrame(
                [
                    ["1_0_0_u", "1_1_0_u", np.nan],
                    [np.nan, np.nan, np.nan],
                    ["1_0_2_u", "1_1_2_u", np.nan],
                ],
                index=updated_index[3:],
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask)
            .data[0],
            pd.DataFrame(
                [["0_1_1"], ["0_1_2"], ["0_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask)
            .data[1],
            pd.DataFrame(
                [["1_1_1"], ["1_1_2"], ["1_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data[0],
            pd.DataFrame(
                [],
                index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None),
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(index_mask=update_index_mask, concat=False)
            .data[1],
            pd.DataFrame(
                [],
                index=pd.DatetimeIndex([], dtype="datetime64[ns, UTC]", freq=None),
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data[0],
            pd.DataFrame(
                [["0_1_1"], ["0_1_2"], ["0_1_3"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2)
            .data[1],
            pd.DataFrame(
                [["1_1_1"], ["1_1_2"], ["1_1_0_u"]],
                index=index[1:4],
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data[0],
            pd.DataFrame(
                [["0_1_3"]],
                index=pd.DatetimeIndex(["2020-01-04 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None),
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(
                [0, 1],
                shape=(5, 3),
                index_mask=index_mask,
                column_mask=column_mask,
                missing_index="drop",
                missing_columns="drop",
            )
            .update(n=3, index_mask=update_index_mask2, concat=False)
            .data[1],
            pd.DataFrame(
                [["1_1_0_u"]],
                index=pd.DatetimeIndex(["2020-01-04 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None),
                columns=pd.Index([1], dtype="int64"),
            ),
        )
        assert MyData.fetch(
            [0, 1],
            shape=(5, 3),
            index_mask=index_mask,
            column_mask=column_mask,
            missing_index="drop",
            missing_columns="drop",
        ).last_index == {0: index[4], 1: index[3]}
        assert MyData.fetch(
            [0, 1],
            shape=(5, 3),
            index_mask=index_mask,
            column_mask=column_mask,
            missing_index="drop",
            missing_columns="drop",
        ).update(n=3, index_mask=update_index_mask2).last_index == {0: updated_index[4], 1: updated_index[5]}
        assert MyData.fetch(
            [0, 1],
            shape=(5, 3),
            index_mask=index_mask,
            column_mask=column_mask,
            missing_index="drop",
            missing_columns="drop",
        ).update(n=3, index_mask=update_index_mask2, concat=False).last_index == {
            0: updated_index[4],
            1: updated_index[5],
        }
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3)).update(n=2, return_none=vbt.symbol_dict({0: True, 1: False})).data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3)).update(n=2, return_none=vbt.symbol_dict({0: True, 1: False})).data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", "1_2_0"],
                    ["1_0_1", "1_1_1", "1_2_1"],
                    ["1_0_2", "1_1_2", "1_2_2"],
                    ["1_0_3", "1_1_3", "1_2_3"],
                    ["1_0_0_u", "1_1_0_u", "1_2_0_u"],
                    ["1_0_1_u", "1_1_1_u", "1_2_1_u"],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3))
            .update(n=2, return_none=vbt.symbol_dict({0: True, 1: False}), concat=False)
            .data[0],
            pd.DataFrame(
                [["0_0_4", "0_1_4", "0_2_4"], [np.nan, np.nan, np.nan]],
                index=pd.DatetimeIndex(
                    ["2020-01-05 00:00:00+00:00", "2020-01-06 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3))
            .update(n=2, return_none=vbt.symbol_dict({0: True, 1: False}), concat=False)
            .data[1],
            pd.DataFrame(
                [["1_0_0_u", "1_1_0_u", "1_2_0_u"], ["1_0_1_u", "1_1_1_u", "1_2_1_u"]],
                index=pd.DatetimeIndex(
                    ["2020-01-05 00:00:00+00:00", "2020-01-06 00:00:00+00:00"], dtype="datetime64[ns, UTC]", freq=None
                ),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3)).update(n=2, return_empty=vbt.symbol_dict({0: True, 1: False})).data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3)).update(n=2, return_empty=vbt.symbol_dict({0: True, 1: False})).data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", "1_2_0"],
                    ["1_0_1", "1_1_1", "1_2_1"],
                    ["1_0_2", "1_1_2", "1_2_2"],
                    ["1_0_3", "1_1_3", "1_2_3"],
                    ["1_0_0_u", "1_1_0_u", "1_2_0_u"],
                    ["1_0_1_u", "1_1_1_u", "1_2_1_u"],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3))
            .update(n=2, raise_error=vbt.symbol_dict({0: True, 1: False}), skip_on_error=True)
            .data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                    [np.nan, np.nan, np.nan],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3))
            .update(n=2, raise_error=vbt.symbol_dict({0: True, 1: False}), skip_on_error=True)
            .data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", "1_2_0"],
                    ["1_0_1", "1_1_1", "1_2_1"],
                    ["1_0_2", "1_1_2", "1_2_2"],
                    ["1_0_3", "1_1_3", "1_2_3"],
                    ["1_0_0_u", "1_1_0_u", "1_2_0_u"],
                    ["1_0_1_u", "1_1_1_u", "1_2_1_u"],
                ],
                index=updated_index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3)).update(n=2, raise_error=True, skip_on_error=True).data[0],
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                ],
                index=index,
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3)).update(n=2, raise_error=True, skip_on_error=True).data[1],
            pd.DataFrame(
                [
                    ["1_0_0", "1_1_0", "1_2_0"],
                    ["1_0_1", "1_1_1", "1_2_1"],
                    ["1_0_2", "1_1_2", "1_2_2"],
                    ["1_0_3", "1_1_3", "1_2_3"],
                    ["1_0_4", "1_1_4", "1_2_4"],
                ],
                index=index,
            ),
        )

    def test_concat(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), columns="feat0").concat()["feat0"],
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_4"], index=index, name=0),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5,), columns="feat0").concat()["feat0"],
            pd.DataFrame(
                [["0_0", "1_0"], ["0_1", "1_1"], ["0_2", "1_2"], ["0_3", "1_3"], ["0_4", "1_4"]],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat0"],
            pd.Series(["0_0_0", "0_0_1", "0_0_2", "0_0_3", "0_0_4"], index=index, name=0),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat1"],
            pd.Series(["0_1_0", "0_1_1", "0_1_2", "0_1_3", "0_1_4"], index=index, name=0),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat2"],
            pd.Series(["0_2_0", "0_2_1", "0_2_2", "0_2_3", "0_2_4"], index=index, name=0),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat0"],
            pd.DataFrame(
                [["0_0_0", "1_0_0"], ["0_0_1", "1_0_1"], ["0_0_2", "1_0_2"], ["0_0_3", "1_0_3"], ["0_0_4", "1_0_4"]],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat1"],
            pd.DataFrame(
                [["0_1_0", "1_1_0"], ["0_1_1", "1_1_1"], ["0_1_2", "1_1_2"], ["0_1_3", "1_1_3"], ["0_1_4", "1_1_4"]],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).concat()["feat2"],
            pd.DataFrame(
                [["0_2_0", "1_2_0"], ["0_2_1", "1_2_1"], ["0_2_2", "1_2_2"], ["0_2_3", "1_2_3"], ["0_2_4", "1_2_4"]],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )

    def test_get(self):
        index = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00",
                "2020-01-02 00:00:00",
                "2020-01-03 00:00:00",
                "2020-01-04 00:00:00",
                "2020-01-05 00:00:00",
            ],
            freq="D",
            tz=timezone.utc,
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5,), columns="feat0").get(),
            pd.Series(["0_0", "0_1", "0_2", "0_3", "0_4"], index=index, name="feat0"),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(),
            pd.DataFrame(
                [
                    ["0_0_0", "0_1_0", "0_2_0"],
                    ["0_0_1", "0_1_1", "0_2_1"],
                    ["0_0_2", "0_1_2", "0_2_2"],
                    ["0_0_3", "0_1_3", "0_2_3"],
                    ["0_0_4", "0_1_4", "0_2_4"],
                ],
                index=index,
                columns=pd.Index(["feat0", "feat1", "feat2"], dtype="object"),
            ),
        )
        pd.testing.assert_series_equal(
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get("feat0"),
            pd.Series(["0_0_0", "0_0_1", "0_0_2", "0_0_3", "0_0_4"], index=index, name="feat0"),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5,), columns="feat0").get(),
            pd.DataFrame(
                [["0_0", "1_0"], ["0_1", "1_1"], ["0_2", "1_2"], ["0_3", "1_3"], ["0_4", "1_4"]],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get("feat0"),
            pd.DataFrame(
                [["0_0_0", "1_0_0"], ["0_0_1", "1_0_1"], ["0_0_2", "1_0_2"], ["0_0_3", "1_0_3"], ["0_0_4", "1_0_4"]],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(["feat0", "feat1"])[0],
            pd.DataFrame(
                [["0_0_0", "1_0_0"], ["0_0_1", "1_0_1"], ["0_0_2", "1_0_2"], ["0_0_3", "1_0_3"], ["0_0_4", "1_0_4"]],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get()[0],
            pd.DataFrame(
                [["0_0_0", "1_0_0"], ["0_0_1", "1_0_1"], ["0_0_2", "1_0_2"], ["0_0_3", "1_0_3"], ["0_0_4", "1_0_4"]],
                index=index,
                columns=pd.Index([0, 1], dtype="int64", name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(symbols=0),
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(symbols=[0])[0],
            MyData.fetch([0], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get()[0],
        )
        pd.testing.assert_series_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get("feat0", symbols=0),
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get("feat0"),
        )
        pd.testing.assert_frame_equal(
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(["feat0"], symbols=0),
            MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"]).get(["feat0"]),
        )

    def test_select(self):
        data = MyData.fetch([0, 1, 2], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert data.select(0) == MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert data.select([0]) == MyData.fetch([0], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert data.select([0]) != MyData.fetch(0, shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert data.select([0, 2]) == MyData.fetch([0, 2], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert data.select([0, 2]) != MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])

    def test_rename(self):
        data = MyData.fetch([0, 1, 2], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        renamed_data = data.rename({0: 2, 2: 0})
        assert renamed_data.symbols == [2, 1, 0]
        assert list(renamed_data.data.keys()) == [2, 1, 0]
        assert list(renamed_data.fetch_kwargs.keys()) == [2, 1, 0]
        assert list(renamed_data.returned_kwargs.keys()) == [2, 1, 0]
        assert list(renamed_data.last_index.keys()) == [2, 1, 0]

    def test_merge(self):
        data = MyData.fetch([0, 1, 2], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        data0 = MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        data12 = MyData.fetch([2], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        assert MyData.merge(data0, data12) == data

    def test_to_csv(self, tmp_path):
        data = MyData.fetch(["S1", "S2"], shape=(5, 3), columns=["feat0", "feat1", "feat2"])

        def _load_and_check_symbol(s, path, **kwargs):
            df = pd.read_csv(path, parse_dates=True, index_col=0, **kwargs).squeeze("columns")
            df.index.freq = df.index.inferred_freq
            pd.testing.assert_frame_equal(df, data.data[s])

        data.to_csv(tmp_path)
        _load_and_check_symbol("S1", tmp_path / "S1.csv")
        _load_and_check_symbol("S2", tmp_path / "S2.csv")

        data.to_csv(
            vbt.symbol_dict({"S1": tmp_path / "csv_data", "S2": tmp_path / "csv_data"}),
            ext=vbt.symbol_dict({"S1": "csv", "S2": "tsv"}),
            sep=vbt.symbol_dict({"S1": ",", "S2": "\t"}),
            mkdir_kwargs=dict(mkdir=True),
        )
        _load_and_check_symbol("S1", tmp_path / "csv_data/S1.csv", sep=",")
        _load_and_check_symbol("S2", tmp_path / "csv_data/S2.tsv", sep="\t")

        data.to_csv(path_or_buf=vbt.symbol_dict({"S1": tmp_path / "my_S1.csv", "S2": tmp_path / "my_S2.csv"}))
        _load_and_check_symbol("S1", tmp_path / "my_S1.csv")
        _load_and_check_symbol("S2", tmp_path / "my_S2.csv")

        data.to_csv(
            vbt.symbol_dict({"S1": tmp_path / "csv_data", "S2": tmp_path / "csv_data"}),
            ext=vbt.symbol_dict({"S1": "csv", "S2": "tsv"}),
            sep=vbt.symbol_dict({"S1": ",", "S2": "\t"}),
        )
        _load_and_check_symbol("S1", tmp_path / "csv_data/S1.csv", sep=",")
        _load_and_check_symbol("S2", tmp_path / "csv_data/S2.tsv", sep="\t")

    def test_to_hdf(self, tmp_path):
        data = MyData.fetch(["S1", "S2"], shape=(5, 3), columns=["feat0", "feat1", "feat2"])

        def _load_and_check_symbol(s, path, key=None, **kwargs):
            if key is None:
                key = s
            df = pd.read_hdf(path, key, **kwargs)
            df.index.freq = df.index.inferred_freq
            pd.testing.assert_frame_equal(df, data.data[s])

        data.to_hdf(tmp_path)
        _load_and_check_symbol("S1", tmp_path / "MyData.h5")
        _load_and_check_symbol("S2", tmp_path / "MyData.h5")

        data.to_hdf(
            vbt.symbol_dict({"S1": tmp_path / "hdf_data/S1.h5", "S2": tmp_path / "hdf_data/S2.h5"}),
            mkdir_kwargs=dict(mkdir=True),
        )
        _load_and_check_symbol("S1", tmp_path / "hdf_data/S1.h5")
        _load_and_check_symbol("S2", tmp_path / "hdf_data/S2.h5")

        data.to_hdf(
            vbt.symbol_dict({"S1": tmp_path / "hdf_data/my_data.h5", "S2": tmp_path / "hdf_data/my_data.h5"}),
            key=vbt.symbol_dict({"S1": "df1", "S2": "df2"}),
        )
        _load_and_check_symbol("S1", tmp_path / "hdf_data/my_data.h5", key="df1")
        _load_and_check_symbol("S2", tmp_path / "hdf_data/my_data.h5", key="df2")

        data.to_hdf(
            path_or_buf=vbt.symbol_dict(
                {"S1": tmp_path / "hdf_data/my_data.h5", "S2": tmp_path / "hdf_data/my_data.h5"},
            ),
            key=vbt.symbol_dict({"S1": "df1", "S2": "df2"}),
        )
        _load_and_check_symbol("S1", tmp_path / "hdf_data/my_data.h5", key="df1")
        _load_and_check_symbol("S2", tmp_path / "hdf_data/my_data.h5", key="df2")

    def test_indexing(self):
        assert (
            MyData.fetch([0, 1], shape=(5,), columns="feat0").iloc[:3].wrapper
            == MyData.fetch([0, 1], shape=(3,), columns="feat0").wrapper
        )
        assert (
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"]).iloc[:3].wrapper
            == MyData.fetch([0, 1], shape=(3, 3), columns=["feat0", "feat1", "feat2"]).wrapper
        )
        assert (
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])["feat0"].wrapper
            == MyData.fetch([0, 1], shape=(5,), columns="feat0").wrapper
        )
        assert (
            MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])[["feat0"]].wrapper
            == MyData.fetch([0, 1], shape=(5, 1), columns=["feat0"]).wrapper
        )

    def test_stats(self):
        index_mask = vbt.symbol_dict({0: [False, True, True, True, True], 1: [True, True, True, True, False]})
        column_mask = vbt.symbol_dict({0: [False, True, True], 1: [True, True, False]})
        data = MyData.fetch(
            [0, 1],
            shape=(5, 3),
            index_mask=index_mask,
            column_mask=column_mask,
            missing_index="nan",
            missing_columns="nan",
            columns=["feat0", "feat1", "feat2"],
        )

        stats_index = pd.Index(
            [
                "Start",
                "End",
                "Period",
                "Total Symbols",
                "Last Index: 0",
                "Last Index: 1",
                "Null Counts: 0",
                "Null Counts: 1",
            ],
            dtype="object",
        )
        pd.testing.assert_series_equal(
            data.stats(),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-04 00:00:00+0000", tz="UTC"),
                    7,
                    7,
                ],
                index=stats_index,
                name="agg_stats",
            ),
        )
        pd.testing.assert_series_equal(
            data.stats(column="feat0"),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-04 00:00:00+0000", tz="UTC"),
                    5,
                    1,
                ],
                index=stats_index,
                name="feat0",
            ),
        )
        pd.testing.assert_series_equal(
            data.stats(group_by=True),
            pd.Series(
                [
                    pd.Timestamp("2020-01-01 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    pd.Timestamp("2020-01-05 00:00:00+0000", tz="UTC"),
                    pd.Timestamp("2020-01-04 00:00:00+0000", tz="UTC"),
                    7,
                    7,
                ],
                index=stats_index,
                name="group",
            ),
        )
        pd.testing.assert_series_equal(data["feat0"].stats(), data.stats(column="feat0"))
        pd.testing.assert_series_equal(
            data.replace(wrapper=data.wrapper.replace(group_by=True)).stats(),
            data.stats(group_by=True),
        )
        stats_df = data.stats(agg_func=None)
        assert stats_df.shape == (3, 8)
        pd.testing.assert_index_equal(stats_df.index, data.wrapper.columns)
        pd.testing.assert_index_equal(stats_df.columns, stats_index)

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        ohlcv_data = vbt.Data.from_data(
            vbt.symbol_dict(
                S1=pd.DataFrame(
                    {
                        "Open": [1, 2, 3, 4, 5],
                        "High": [2.5, 3.5, 4.5, 5.5, 6.5],
                        "Low": [0.5, 1.5, 2.5, 3.5, 4.5],
                        "Close": [2, 3, 4, 5, 6],
                        "Volume": [1, 2, 3, 2, 1],
                        "Other": [3, 2, 1, 2, 3],
                    },
                    index=pd.date_range("2020-01-01", "2020-01-05"),
                )
            ),
            single_symbol=True,
        )
        ohlcv_data.column_config["Other"] = dict(
            resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(resampler, vbt.nb.mean_reduce_nb)
        )
        pd.testing.assert_frame_equal(
            ohlcv_data.resample(test_freq).get(),
            pd.concat((
                ohlcv_data.get(["Open", "High", "Low", "Close", "Volume"]).vbt.ohlcv.resample(test_freq).obj,
                ohlcv_data.get(["Other"]).resample(test_freq).mean(),
            ), axis=1)
        )


# ############# custom.py ############# #


class TestCustom:
    def test_csv_data(self, tmp_path):
        sr = pd.Series(np.arange(10))
        sr.to_csv(tmp_path / "temp.csv")
        csv_data = vbt.CSVData.fetch(tmp_path / "temp.csv")
        pd.testing.assert_series_equal(csv_data.get(), sr)
        csv_data = vbt.CSVData.fetch("TEMP", paths=tmp_path / "temp.csv")
        assert csv_data.symbols[0] == "TEMP"
        pd.testing.assert_series_equal(csv_data.get(), sr)
        csv_data = vbt.CSVData.fetch("TEMP", paths=[tmp_path / "temp.csv"])
        assert csv_data.symbols[0] == "TEMP"
        pd.testing.assert_series_equal(csv_data.get(), sr)
        csv_data = vbt.CSVData.fetch(["TEMP"], paths=tmp_path / "temp.csv")
        assert csv_data.symbols[0] == "TEMP"
        pd.testing.assert_series_equal(csv_data.get()["TEMP"], sr.rename("TEMP"))
        csv_data = vbt.CSVData.fetch(tmp_path / "temp.csv", start_row=2, end_row=3)
        pd.testing.assert_series_equal(csv_data.get(), sr.iloc[2:4])
        df = pd.DataFrame(np.arange(20).reshape((10, 2)))
        df.columns = pd.Index(["0", "1"], dtype="object")
        df.to_csv(tmp_path / "temp.csv")
        csv_data = vbt.CSVData.fetch(tmp_path / "temp.csv", iterator=True)
        pd.testing.assert_frame_equal(csv_data.get(), df)
        csv_data = vbt.CSVData.fetch(tmp_path / "temp.csv", chunksize=1)
        pd.testing.assert_frame_equal(csv_data.get(), df)
        csv_data = vbt.CSVData.fetch(tmp_path / "temp.csv", chunksize=1, chunk_func=lambda x: list(x)[-1])
        pd.testing.assert_frame_equal(csv_data.get(), df.iloc[[-1]])
        df = pd.DataFrame(np.arange(20).reshape((10, 2)))
        df.columns = pd.MultiIndex.from_tuples([("1", "2"), ("3", "4")], names=["a", "b"])
        df.to_csv(tmp_path / "temp.csv")
        csv_data = vbt.CSVData.fetch(tmp_path / "temp.csv", header=[0, 1], start_row=0, end_row=1)
        pd.testing.assert_frame_equal(csv_data.get(), df.iloc[:2])
        assert csv_data.returned_kwargs["temp"] == {"last_row": 1}
        csv_data = csv_data.update()
        pd.testing.assert_frame_equal(csv_data.get(), df.iloc[:2])
        assert csv_data.returned_kwargs["temp"] == {"last_row": 1}
        csv_data = csv_data.update(end_row=2)
        csv_data.get()
        pd.testing.assert_frame_equal(csv_data.get(), df.iloc[:3])
        assert csv_data.returned_kwargs["temp"] == {"last_row": 2}
        csv_data = csv_data.update(end_row=None)
        pd.testing.assert_frame_equal(csv_data.get(), df)
        assert csv_data.returned_kwargs["temp"] == {"last_row": 9}

        data1 = MyData.fetch(shape=(5,))
        data2 = MyData.fetch(shape=(6,))
        data3 = MyData.fetch(shape=(7,))
        result_data = vbt.Data.from_data({"data1": data1.get(), "data2": data2.get(), "data3": data3.get()})

        data1.get().to_csv(tmp_path / "data1.csv")
        data2.get().to_csv(tmp_path / "data2.csv")
        data3.get().to_csv(tmp_path / "data3.csv")
        csv_data = vbt.CSVData.fetch(tmp_path / "data*.csv")
        pd.testing.assert_frame_equal(csv_data.get(), result_data.get())
        (tmp_path / "data").mkdir(exist_ok=True)
        data1.get().to_csv(tmp_path / "data/data1.csv")
        data2.get().to_csv(tmp_path / "data/data2.csv")
        data3.get().to_csv(tmp_path / "data/data3.csv")
        csv_data = vbt.CSVData.fetch(tmp_path / "data")
        pd.testing.assert_frame_equal(csv_data.get(), result_data.get())
        csv_data = vbt.CSVData.fetch(
            [tmp_path / "data/data1.csv", tmp_path / "data/data2.csv", tmp_path / "data/data3.csv"],
        )
        pd.testing.assert_frame_equal(csv_data.get(), result_data.get())
        csv_data = vbt.CSVData.fetch(
            paths=[tmp_path / "data/data1.csv", tmp_path / "data/data2.csv", tmp_path / "data/data3.csv"],
        )
        pd.testing.assert_frame_equal(csv_data.get(), result_data.get())
        csv_data = vbt.CSVData.fetch(
            symbols=["DATA1", "DATA2", "DATA3"],
            paths=[tmp_path / "data/data1.csv", tmp_path / "data/data2.csv", tmp_path / "data/data3.csv"],
        )
        pd.testing.assert_frame_equal(
            csv_data.get(),
            result_data.get().rename(columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}),
        )
        csv_data = vbt.CSVData.fetch(
            vbt.symbol_dict(
                {
                    "DATA1": tmp_path / "data/data1.csv",
                    "DATA2": tmp_path / "data/data2.csv",
                    "DATA3": tmp_path / "data/data3.csv",
                }
            )
        )
        pd.testing.assert_frame_equal(
            csv_data.get(),
            result_data.get().rename(columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}),
        )
        with pytest.raises(Exception):
            vbt.CSVData.fetch("DATA")
        with pytest.raises(Exception):
            vbt.CSVData.fetch("DATA", paths=tmp_path / "data/data*.csv")
        with pytest.raises(Exception):
            vbt.CSVData.fetch(["DATA1", "DATA2"], paths=tmp_path / "data/data1.csv")
        with pytest.raises(Exception):
            vbt.CSVData.fetch(None)
        with pytest.raises(Exception):
            vbt.CSVData.fetch(0)

    def test_hdf_data(self, tmp_path):
        sr = pd.Series(np.arange(10))
        sr.to_hdf(tmp_path / "temp.h5", "s")
        hdf_data = vbt.HDFData.fetch(tmp_path / "temp.h5" / "s")
        pd.testing.assert_series_equal(hdf_data.get(), sr)
        hdf_data = vbt.HDFData.fetch("S", paths=tmp_path / "temp.h5" / "s")
        assert hdf_data.symbols[0] == "S"
        pd.testing.assert_series_equal(hdf_data.get(), sr)
        hdf_data = vbt.HDFData.fetch("S", paths=[tmp_path / "temp.h5" / "s"])
        assert hdf_data.symbols[0] == "S"
        pd.testing.assert_series_equal(hdf_data.get(), sr)
        hdf_data = vbt.HDFData.fetch(["S"], paths=tmp_path / "temp.h5" / "s")
        assert hdf_data.symbols[0] == "S"
        pd.testing.assert_series_equal(hdf_data.get()["S"], sr.rename("S"))
        hdf_data = vbt.HDFData.fetch(tmp_path / "temp.h5" / "s", start_row=2, end_row=3)
        pd.testing.assert_series_equal(hdf_data.get(), sr.iloc[2:4])
        df = pd.DataFrame(np.arange(20).reshape((10, 2)))
        df.columns = pd.Index(["0", "1"], dtype="object")
        df.to_hdf(tmp_path / "temp.h5", "df", format="table")
        hdf_data = vbt.HDFData.fetch(tmp_path / "temp.h5" / "df", iterator=True)
        pd.testing.assert_frame_equal(hdf_data.get(), df)
        hdf_data = vbt.HDFData.fetch(tmp_path / "temp.h5" / "df", chunksize=1)
        pd.testing.assert_frame_equal(hdf_data.get(), df)
        hdf_data = vbt.HDFData.fetch(tmp_path / "temp.h5" / "df", chunksize=1, chunk_func=lambda x: list(x)[-1])
        pd.testing.assert_frame_equal(hdf_data.get(), df.iloc[[-1]])
        df = pd.DataFrame(np.arange(20).reshape((10, 2)))
        df.columns = pd.MultiIndex.from_tuples([("1", "2"), ("3", "4")], names=["a", "b"])
        df.to_hdf(tmp_path / "temp.h5", "df")
        hdf_data = vbt.HDFData.fetch(tmp_path / "temp.h5" / "df", header=[0, 1], start_row=0, end_row=1)
        pd.testing.assert_frame_equal(hdf_data.get(), df.iloc[:2])
        assert hdf_data.returned_kwargs["df"] == {"last_row": 1}
        hdf_data = hdf_data.update()
        pd.testing.assert_frame_equal(hdf_data.get(), df.iloc[:2])
        assert hdf_data.returned_kwargs["df"] == {"last_row": 1}
        hdf_data = hdf_data.update(end_row=2)
        hdf_data.get()
        pd.testing.assert_frame_equal(hdf_data.get(), df.iloc[:3])
        assert hdf_data.returned_kwargs["df"] == {"last_row": 2}
        hdf_data = hdf_data.update(end_row=None)
        pd.testing.assert_frame_equal(hdf_data.get(), df)
        assert hdf_data.returned_kwargs["df"] == {"last_row": 9}

        data1 = MyData.fetch(shape=(5,))
        data2 = MyData.fetch(shape=(6,))
        data3 = MyData.fetch(shape=(7,))
        result_data = vbt.Data.from_data({"data1": data1.get(), "data2": data2.get(), "data3": data3.get()})

        data1.get().to_hdf(tmp_path / "data1.h5", "data1")
        data2.get().to_hdf(tmp_path / "data2.h5", "data2")
        data3.get().to_hdf(tmp_path / "data3.h5", "data3")
        hdf_data = vbt.HDFData.fetch(tmp_path / "data*.h5")
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())
        (tmp_path / "data").mkdir(exist_ok=True)
        data1.get().to_hdf(tmp_path / "data/data1.h5", "data1")
        data2.get().to_hdf(tmp_path / "data/data2.h5", "data2")
        data3.get().to_hdf(tmp_path / "data/data3.h5", "data3")
        hdf_data = vbt.HDFData.fetch(tmp_path / "data")
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())
        hdf_data = vbt.HDFData.fetch(
            [tmp_path / "data/data1.h5", tmp_path / "data/data2.h5", tmp_path / "data/data3.h5"],
        )
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())
        hdf_data = vbt.HDFData.fetch(
            paths=[tmp_path / "data/data1.h5", tmp_path / "data/data2.h5", tmp_path / "data/data3.h5"],
        )
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())
        hdf_data = vbt.HDFData.fetch(
            symbols=["DATA1", "DATA2", "DATA3"],
            paths=[tmp_path / "data/data1.h5", tmp_path / "data/data2.h5", tmp_path / "data/data3.h5"],
        )
        pd.testing.assert_frame_equal(
            hdf_data.get(),
            result_data.get().rename(columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}),
        )
        hdf_data = vbt.HDFData.fetch(
            vbt.symbol_dict(
                {
                    "DATA1": tmp_path / "data/data1.h5",
                    "DATA2": tmp_path / "data/data2.h5",
                    "DATA3": tmp_path / "data/data3.h5",
                }
            )
        )
        pd.testing.assert_frame_equal(
            hdf_data.get(),
            result_data.get().rename(columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}),
        )
        with pytest.raises(Exception):
            vbt.HDFData.fetch("DATA")
        with pytest.raises(Exception):
            vbt.HDFData.fetch("DATA", paths=tmp_path / "data/data*.h5")
        with pytest.raises(Exception):
            vbt.HDFData.fetch(["DATA1", "DATA2"], paths=tmp_path / "data/data1.h5")
        with pytest.raises(Exception):
            vbt.HDFData.fetch(None)
        with pytest.raises(Exception):
            vbt.HDFData.fetch(0)

        data1.get().to_hdf(tmp_path / "data/data.h5", "data1")
        data2.get().to_hdf(tmp_path / "data/data.h5", "data2")
        data3.get().to_hdf(tmp_path / "data/data.h5", "data3")
        hdf_data = vbt.HDFData.fetch(tmp_path / "data/data.h5")
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())
        data1.get().to_hdf(tmp_path / "data/data.h5", "data1")
        data2.get().to_hdf(tmp_path / "data/data.h5", "data2")
        data3.get().to_hdf(tmp_path / "data/data.h5", "data3")
        hdf_data = vbt.HDFData.fetch(tmp_path / "data")
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())
        data1.get().to_hdf(tmp_path / "data/data.h5", "/folder/data1")
        data2.get().to_hdf(tmp_path / "data/data.h5", "/folder/data2")
        data3.get().to_hdf(tmp_path / "data/data.h5", "/folder/data3")
        hdf_data = vbt.HDFData.fetch(tmp_path / "data/data.h5/folder")
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())
        hdf_data = vbt.HDFData.fetch(
            symbols=["DATA1", "DATA2", "DATA3"],
            paths=[
                tmp_path / "data/data.h5/folder/data1",
                tmp_path / "data/data.h5/folder/data2",
                tmp_path / "data/data.h5/folder/data3",
            ],
        )
        pd.testing.assert_frame_equal(
            hdf_data.get(),
            result_data.get().rename(columns={"data1": "DATA1", "data2": "DATA2", "data3": "DATA3"}),
        )
        if (tmp_path / "data/data.h5").exists():
            (tmp_path / "data/data.h5").unlink()
        data1.get().to_hdf(tmp_path / "data/data.h5", "/data1/folder")
        data2.get().to_hdf(tmp_path / "data/data.h5", "/data2/folder")
        data3.get().to_hdf(tmp_path / "data/data.h5", "/data3/folder")
        with pytest.raises(Exception):
            vbt.HDFData.fetch(tmp_path / "data/data.h5/folder")

        if (tmp_path / "data/data.h5").exists():
            (tmp_path / "data/data.h5").unlink()
        data1.get().to_hdf(tmp_path / "data/data.h5", "/data1/folder/data1")
        data2.get().to_hdf(tmp_path / "data/data.h5", "/data2/folder/data2")
        data3.get().to_hdf(tmp_path / "data/data.h5", "/data3/folder/data3")
        hdf_data = vbt.HDFData.fetch(tmp_path / "data/data.h5/data*/folder/*")
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())

        if (tmp_path / "data/data.h5").exists():
            (tmp_path / "data/data.h5").unlink()
        data1.get().to_hdf(tmp_path / "data/data.h5", "/data1/folder/data1")
        data2.get().to_hdf(tmp_path / "data/data.h5", "/data2/folder/data2")
        data3.get().to_hdf(tmp_path / "data/data.h5", "/data3/folder/data3")
        hdf_data = vbt.HDFData.fetch(tmp_path / "**/data.h5/data*/folder/*")
        pd.testing.assert_frame_equal(hdf_data.get(), result_data.get())

        with pytest.raises(Exception):
            vbt.HDFData.fetch(tmp_path / "data/data.h5/folder/data4")

    def test_random_data(self):
        pd.testing.assert_series_equal(
            vbt.RandomData.fetch(start="2021-01-01 UTC", end="2021-01-05 UTC", seed=42).get(),
            pd.Series(
                [100.49671415301123, 100.35776307348756, 101.00776880200878, 102.54614727815496, 102.3060320136544],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
            ),
        )
        pd.testing.assert_series_equal(
            vbt.RandomData.fetch(start="2021-01-01 UTC", end="2021-01-05 UTC", symmetric=True, seed=42).get(),
            pd.Series(
                [100.49671415301123, 100.35795492796039, 101.00796189910105, 102.54634331617359, 102.30678851828695],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
            ),
        )
        pd.testing.assert_frame_equal(
            vbt.RandomData.fetch(num_paths=2, start="2021-01-01 UTC", end="2021-01-05 UTC", seed=42).get(),
            pd.DataFrame(
                [
                    [100.49671415301123, 99.7658630430508],
                    [100.35776307348756, 101.34137833772823],
                    [101.00776880200878, 102.11910727009419],
                    [102.54614727815496, 101.63968421831567],
                    [102.3060320136544, 102.1911405333112],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
                columns=pd.Index([0, 1], name="path"),
            ),
        )
        pd.testing.assert_frame_equal(
            vbt.RandomData.fetch([0, 1], start="2021-01-01 UTC", end="2021-01-05 UTC", seed=42).get(),
            pd.DataFrame(
                [
                    [100.49671415301123, 100.49671415301123],
                    [100.35776307348756, 100.35776307348756],
                    [101.00776880200878, 101.00776880200878],
                    [102.54614727815496, 102.54614727815496],
                    [102.3060320136544, 102.3060320136544],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
                columns=pd.Index([0, 1], name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            vbt.RandomData.fetch(
                start="2021-01-01 UTC",
                end="2021-01-05 UTC",
                to_ohlc=True,
                ohlc_freq="2D",
                seed=42,
            ).get(),
            pd.DataFrame(
                [
                    [100.49671415301123, 100.49671415301123, 100.35776307348756, 100.35776307348756],
                    [101.00776880200878, 102.54614727815496, 101.00776880200878, 102.54614727815496],
                    [102.3060320136544, 102.3060320136544, 102.3060320136544, 102.3060320136544],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="2D",
                ),
                columns=pd.Index(['Open', 'High', 'Low', 'Close'], dtype='object'),
            ),
        )

    def test_gbm_data(self):
        pd.testing.assert_series_equal(
            vbt.GBMData.fetch(start="2021-01-01 UTC", end="2021-01-05 UTC", seed=42).get(),
            pd.Series(
                [100.49292505095792, 100.34905764408163, 100.99606643427086, 102.54091282498935, 102.29597577584751],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
            ),
        )
        pd.testing.assert_frame_equal(
            vbt.GBMData.fetch(num_paths=2, start="2021-01-01 UTC", end="2021-01-05 UTC", seed=42).get(),
            pd.DataFrame(
                [
                    [100.49292505095792, 99.76114874768454],
                    [100.34905764408163, 101.34402779029647],
                    [100.99606643427086, 102.119662952671],
                    [102.54091282498935, 101.6362789823718],
                    [102.29597577584751, 102.1841061387023],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
                columns=pd.Index([0, 1], name="path"),
            ),
        )
        pd.testing.assert_frame_equal(
            vbt.GBMData.fetch([0, 1], start="2021-01-01 UTC", end="2021-01-05 UTC", seed=42).get(),
            pd.DataFrame(
                [
                    [100.49292505095792, 100.49292505095792],
                    [100.34905764408163, 100.34905764408163],
                    [100.99606643427086, 100.99606643427086],
                    [102.54091282498935, 102.54091282498935],
                    [102.29597577584751, 102.29597577584751],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 2),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 4),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="D",
                ),
                columns=pd.Index([0, 1], name="symbol"),
            ),
        )
        pd.testing.assert_frame_equal(
            vbt.GBMData.fetch(
                start="2021-01-01 UTC",
                end="2021-01-05 UTC",
                to_ohlc=True,
                ohlc_freq="2D",
                seed=42,
            ).get(),
            pd.DataFrame(
                [
                    [100.49292505095792, 100.49292505095792, 100.34905764408163, 100.34905764408163],
                    [100.99606643427086, 102.54091282498935, 100.99606643427086, 102.54091282498935],
                    [102.29597577584751, 102.29597577584751, 102.29597577584751, 102.29597577584751],
                ],
                index=pd.DatetimeIndex(
                    [
                        datetime(2021, 1, 1),
                        datetime(2021, 1, 3),
                        datetime(2021, 1, 5),
                    ],
                    dtype="datetime64[ns, UTC]",
                    freq="2D",
                ),
                columns=pd.Index(['Open', 'High', 'Low', 'Close'], dtype='object'),
            ),
        )


# ############# updater.py ############# #


class TestDataUpdater:
    def test_update(self):
        data = MyData.fetch(0, shape=(5,), return_arr=True)
        updater = vbt.DataUpdater(data)
        updater.update()
        assert updater.data == data.update()
        assert updater.config["data"] == data.update()

    def test_update_every(self):
        data = MyData.fetch(0, shape=(5,), return_arr=True)
        kwargs = dict(call_count=0)

        class DataUpdater(vbt.DataUpdater):
            def update(self, kwargs):
                super().update()
                kwargs["call_count"] += 1
                if kwargs["call_count"] == 5:
                    raise vbt.CancelledError

        updater = DataUpdater(data)
        updater.update_every(kwargs=kwargs)
        for i in range(5):
            data = data.update()
        assert updater.data == data
        assert updater.config["data"] == data


# ############# saver.py ############# #


class TestCSVDataSaver:
    def test_update(self, tmp_path):
        data = MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        saver = vbt.CSVDataSaver(
            data,
            save_kwargs=dict(
                dir_path=tmp_path / "saver",
                mkdir_kwargs=dict(mkdir=True),
            ),
        )
        saver.init_save_data()
        saver.update(n=2)
        updated_data = data.update(n=2, concat=False)
        assert saver.data == updated_data
        saved_result0 = pd.concat((data.data[0].iloc[:-1], updated_data.data[0]), axis=0)
        saved_result0.index.freq = "D"
        saved_result1 = pd.concat((data.data[1].iloc[:-1], updated_data.data[1]), axis=0)
        saved_result1.index.freq = "D"
        pd.testing.assert_frame_equal(vbt.CSVData.fetch(tmp_path / "saver").data["0"], saved_result0)
        pd.testing.assert_frame_equal(vbt.CSVData.fetch(tmp_path / "saver").data["1"], saved_result1)

        new_data = saver.data
        new_saver = vbt.CSVDataSaver(
            new_data,
            save_kwargs=dict(
                dir_path=tmp_path / "saver",
                mkdir_kwargs=dict(mkdir=True),
            ),
        )
        new_saver.update(n=2)
        new_updated_data = new_data.update(n=2, concat=False)
        assert new_saver.data == new_updated_data
        new_saved_result0 = pd.concat(
            (data.data[0].iloc[:-1], new_data.data[0].iloc[:-1], new_updated_data.data[0]), axis=0
        )
        new_saved_result0.index.freq = "D"
        new_saved_result1 = pd.concat(
            (data.data[1].iloc[:-1], new_data.data[1].iloc[:-1], new_updated_data.data[1]), axis=0
        )
        new_saved_result1.index.freq = "D"
        pd.testing.assert_frame_equal(vbt.CSVData.fetch(tmp_path / "saver").data["0"], new_saved_result0)
        pd.testing.assert_frame_equal(vbt.CSVData.fetch(tmp_path / "saver").data["1"], new_saved_result1)

    def test_update_every(self, tmp_path):
        data = MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        call_count = [0]

        class CSVDataSaver(vbt.CSVDataSaver):
            def update(self, call_count, **kwargs):
                super().update(**kwargs)
                call_count[0] += 1
                if call_count[0] == 5:
                    raise vbt.CancelledError

        saver = CSVDataSaver(
            data,
            save_kwargs=dict(
                dir_path=tmp_path / "saver",
                mkdir_kwargs=dict(mkdir=True),
            ),
        )
        saver.init_save_data()
        saver.update_every(call_count=call_count)
        for i in range(5):
            data = data.update()
        pd.testing.assert_frame_equal(vbt.CSVData.fetch(tmp_path / "saver").data["0"], data.data[0])
        pd.testing.assert_frame_equal(vbt.CSVData.fetch(tmp_path / "saver").data["1"], data.data[1])


class TestHDFDataSaver:
    def test_update(self, tmp_path):
        data = MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        saver = vbt.HDFDataSaver(
            data,
            save_kwargs=dict(
                file_path=tmp_path / "saver.h5",
                mkdir_kwargs=dict(mkdir=True),
                min_itemsize=10,
            ),
        )
        saver.init_save_data()
        saver.update(n=2)
        updated_data = data.update(n=2, concat=False)
        assert saver.data == updated_data
        saved_result0 = pd.concat((data.data[0].iloc[:-1], updated_data.data[0]), axis=0)
        saved_result0.index.freq = "D"
        saved_result1 = pd.concat((data.data[1].iloc[:-1], updated_data.data[1]), axis=0)
        saved_result1.index.freq = "D"
        pd.testing.assert_frame_equal(vbt.HDFData.fetch(tmp_path / "saver.h5").data["0"], saved_result0)
        pd.testing.assert_frame_equal(vbt.HDFData.fetch(tmp_path / "saver.h5").data["1"], saved_result1)

        new_data = saver.data
        new_saver = vbt.HDFDataSaver(
            new_data,
            save_kwargs=dict(
                file_path=tmp_path / "saver.h5",
                mkdir_kwargs=dict(mkdir=True),
                min_itemsize=10,
            ),
        )
        new_saver.update(n=2)
        new_updated_data = new_data.update(n=2, concat=False)
        assert new_saver.data == new_updated_data
        new_saved_result0 = pd.concat(
            (data.data[0].iloc[:-1], new_data.data[0].iloc[:-1], new_updated_data.data[0]), axis=0
        )
        new_saved_result0.index.freq = "D"
        new_saved_result1 = pd.concat(
            (data.data[1].iloc[:-1], new_data.data[1].iloc[:-1], new_updated_data.data[1]), axis=0
        )
        new_saved_result1.index.freq = "D"
        pd.testing.assert_frame_equal(vbt.HDFData.fetch(tmp_path / "saver.h5").data["0"], new_saved_result0)
        pd.testing.assert_frame_equal(vbt.HDFData.fetch(tmp_path / "saver.h5").data["1"], new_saved_result1)

    def test_update_every(self, tmp_path):
        data = MyData.fetch([0, 1], shape=(5, 3), columns=["feat0", "feat1", "feat2"])
        call_count = [0]

        class HDFDataSaver(vbt.HDFDataSaver):
            def update(self, call_count, **kwargs):
                super().update(**kwargs)
                call_count[0] += 1
                if call_count[0] == 5:
                    raise vbt.CancelledError

        saver = HDFDataSaver(
            data,
            save_kwargs=dict(
                file_path=tmp_path / "saver.h5",
                mkdir_kwargs=dict(mkdir=True),
                min_itemsize=10,
            ),
        )
        saver.init_save_data()
        saver.update_every(call_count=call_count)
        for i in range(5):
            data = data.update()
        pd.testing.assert_frame_equal(vbt.HDFData.fetch(tmp_path / "saver.h5").data["0"], data.data[0])
        pd.testing.assert_frame_equal(vbt.HDFData.fetch(tmp_path / "saver.h5").data["1"], data.data[1])
