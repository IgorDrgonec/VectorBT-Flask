import os
from datetime import datetime
import pytest

import numpy as np
import pandas as pd

import vectorbtpro as vbt

ohlcv_ts = pd.DataFrame(
    {
        "open": [1, 2, 3, 4, 5],
        "high": [2.5, 3.5, 4.5, 5.5, 6.5],
        "low": [0.5, 1.5, 2.5, 3.5, 4.5],
        "close": [2, 3, 4, 5, 6],
        "volume": [1, 2, 3, 2, 1],
    },
    index=pd.Index(
        [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4), datetime(2020, 1, 5)],
    ),
)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.chunking["n_chunks"] = 2


def teardown_module():
    vbt.settings.reset()


# ############# accessors.py ############# #


class TestAccessors:
    def test_vwap(self):
        pd.testing.assert_series_equal(
            ohlcv_ts.vbt.ohlcv.vwap(),
            (
                (ohlcv_ts["volume"] * (ohlcv_ts["high"] + ohlcv_ts["low"]) / 2).cumsum() / ohlcv_ts["volume"].cumsum()
            ).rename("vwap"),
        )
        result = pd.concat((ohlcv_ts.vbt.ohlcv.vwap(), ohlcv_ts.vbt.fshift(1).vbt.ohlcv.vwap()), axis=1)
        result.columns = ["a", "b"]
        high = pd.concat((ohlcv_ts["high"], ohlcv_ts["high"].vbt.fshift(1)), axis=1)
        low = pd.concat((ohlcv_ts["low"], ohlcv_ts["low"].vbt.fshift(1)), axis=1)
        volume = pd.concat((ohlcv_ts["volume"], ohlcv_ts["volume"].vbt.fshift(1)), axis=1)
        high.columns = result.columns
        low.columns = result.columns
        volume.columns = result.columns
        pd.testing.assert_frame_equal(pd.DataFrame.vbt.ohlcv.vwap(high=high, low=low, volume=volume), result)

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        pd.testing.assert_frame_equal(
            ohlcv_ts.vbt.ohlcv.resample(test_freq).obj,
            ohlcv_ts.resample(test_freq).agg({
                "open": lambda x: float(x[0] if len(x) > 0 else np.nan),
                "high": lambda x: float(x.max() if len(x) > 0 else np.nan),
                "low": lambda x: float(x.min() if len(x) > 0 else np.nan),
                "close": lambda x: float(x[-1] if len(x) > 0 else np.nan),
                "volume": lambda x: float(x.sum() if len(x) > 0 else np.nan)
            })
        )
