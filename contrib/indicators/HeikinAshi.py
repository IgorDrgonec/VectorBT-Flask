from numba import njit
import numpy as np
import pandas as pd


@njit(cache=True)
def internal_build_ha(open, high, low, close, precision):
    """
    Internal, numba optimized HeikinAshi function
    """
    ha_open = np.full(close.shape[0], np.nan)
    ha_high = np.full(close.shape[0], np.nan)
    ha_low = np.full(close.shape[0], np.nan)
    ha_close = (open + high + low + close) / 4
    np.round(ha_close, precision, ha_close)

    ha_open[0] = open[0]
    ha_high[0] = high[0]
    ha_low[0] = low[0]

    for i in range(ha_open.shape[0]):
        if i > 0:
            ha_open[i] = round((ha_open[i - 1] + ha_close[i - 1]) / 2, precision)
            ha_high[i] = round(max(ha_open[i], ha_close[i], high[i]), precision)
            ha_low[i] = round(min(ha_open[i], ha_close[i], low[i]), precision)

    return ha_open, ha_high, ha_low, ha_close


def HeikinAshi(open, high, low, close, precision=2):
    """
    This HeikinAshi function expects a precision to round the data to.
    This is important to keep the HeikinAshi OHLC consistent with the original input

    This function returns OHLC as 4 different arrays. In addition, it return a dataframe with the HA values.

    AUTHOR: Piotr Yordanov --> https://piotryordanov.com
    """
    ha_nb = internal_build_ha(open, high, low, close, precision)
    ha_df = pd.DataFrame(
        {"Open": ha_nb[0], "High": ha_nb[1], "Low": ha_nb[2], "Close": ha_nb[3]}
    )
    return (
        ha_nb[0],
        ha_nb[1],
        ha_nb[2],
        ha_nb[3],
        ha_df,
    )


### ============================================================================= ###
###                             USAGE
### ============================================================================= ###
def get_precision(prices):
    number = prices[0]
    a = len(str(number).split(".")[1])
    number = prices[1]
    a1 = len(str(number).split(".")[1])
    number = prices[2]
    a2 = len(str(number).split(".")[1])
    precision = max(a, a1, a2)
    return precision


data = vbt.BinanceData.fetch(
    ["BTCUSDT"],
    timeframe="1h",
)
open = np.array(data["Open"])
high = np.array(data["High"])
low = np.array(data["Low"])
close = np.array(data["Close"])
precision = get_precision(close)

HA = HeikinAshi(open, high, low, close, precision)
HA["Close"]
