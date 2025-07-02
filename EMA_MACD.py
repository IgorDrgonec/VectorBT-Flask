import numpy as np
import pandas as pd
import time
import talib
import scipy
import os
import matplotlib.pyplot as plt
from scipy.stats import linregress
from vectorbtpro import *
from binance.client import Client
from datetime import datetime, timedelta
from decimal import Decimal

timeframe = "15m"
csv_file = "ema_macd_data.csv"

if os.path.exists(csv_file):
    try:
        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print("[INFO] Loaded data from CSV")
    except Exception as e:
        print(f"[ERROR] Failed to load cached CSV: {e}")
        data = pd.DataFrame()
else:
    print("[CRITICAL] CSV file not found. Please run init_data.py before starting app.")
    data = pd.DataFrame()

high = data.get('High')
low = data.get('Low')
close = data.get('Close')

#get_SL_TP
@njit(nogil=True)
def get_SL_TP(close, atr, long_entry, short_entry, atr_multiplier = 2, RR = 1.5):
    SL = np.full(close.shape[0], np.nan)
    TP = np.full(close.shape[0], np.nan)
    SL_percentage = np.full(close.shape[0], np.nan)
    TP_percantage = np.full(close.shape[0], np.nan)
    in_trade = "no trade"
    
    long_SL = close - (atr * atr_multiplier)
    long_TP = close + ((atr * atr_multiplier)*RR)
    short_SL = close + (atr * atr_multiplier)
    short_TP = close - ((atr * atr_multiplier)*RR)
    
    for i in range(close.shape[0]):
        if in_trade == "no trade":
            if long_entry[i]:
                SL[i], TP[i] = long_SL[i], long_TP[i]
                in_trade = "long"
            elif short_entry[i]:
                SL[i], TP[i] = short_SL[i], short_TP[i]
                in_trade = "short"
        elif in_trade == "long" and ((close[i] < SL[i-1]) or (close[i]> TP[i-1])):
            in_trade = "no trade"
        elif in_trade == "short" and ((close[i] > SL[i-1]) or (close[i]< TP[i-1])):
            in_trade = "no trade"
        else:
            SL[i], TP[i] = SL[i-1], TP[i-1]

    for i in range(close.shape[0]):
        SL_percentage[i] =  abs(SL[i] - close[i]) / close[i]
        TP_percantage[i] =  abs(TP[i] - close[i]) / close[i]
        
    return SL, TP, SL_percentage, TP_percantage

def ema_macd(high, low, close, atr_multiplier = 2, RR = 1.5, atr_period = 7, ema_window = 200):
    SL = np.full(close.shape[0], np.nan)
    TP = np.full(close.shape[0], np.nan)
    SL_percentage = np.full(close.shape[0], np.nan)
    TP_percantage = np.full(close.shape[0], np.nan)
    uptrend = np.full(close.shape[0], np.nan)
    donwtrend = np.full(close.shape[0], np.nan)
    #Fix our Data
    close = vbt.nb.ffill_1d_nb(close)

    #Calculate indicators
    atr = talib.ATR(high, low, close, atr_period)
    macd, macd_signal, macd_hist = talib.MACD(close)
    ema = talib.EMA(close, ema_window)

    #Calculate signals
    long_entry = (vbt.nb.crossed_above_1d_nb(macd, macd_signal)) & (close > ema) & (macd < 0)
    short_entry = (vbt.nb.crossed_below_1d_nb(macd, macd_signal)) & (close < ema) & (macd > 0)

    #Trend - use TRENDLB
    trend = close > ema

    SL, TP, SL_percentage, TP_percantage = get_SL_TP(close,atr,long_entry,short_entry, atr_multiplier, RR)
    
    return long_entry, short_entry, trend, SL, TP,SL_percentage, TP_percantage, ema

def shift_array(arr):
    shifted_arr = np.roll(arr, 1)
    shifted_arr[0] = np.nan
    return shifted_arr

strat = vbt.IF(
   class_name="EMA MACD",
   short_name='emamcd',
   input_names=['high', 'low', 'close'],
   param_names=[ "atr_multiplier", "RR",'atr_period', 'ema_window'],
   output_names=['long_entry', 'short_entry', 'trend', "SL", "TP","SL_percentage", "TP_percentage", "ema"] 
).with_apply_func(
    ema_macd,
    takes_1d=True,
    atr_multiplier = 2, 
    RR = 1.5,
    atr_period = 14,
    ema_window = 200
)

@njit
def adjust_func_nb(c, size, sl_stop, delta_format, risk_amount):
    close_now = vbt.pf_nb.select_nb(c, c.close)
    sl_stop_now = vbt.pf_nb.select_nb(c, sl_stop)
    delta_format_now = vbt.pf_nb.select_nb(c, delta_format)
    risk_amount_now = vbt.pf_nb.select_nb(c, risk_amount)
    free_cash_now = vbt.pf_nb.get_free_cash_nb(c)

    stop_price = vbt.pf_nb.resolve_stop_price_nb(
        init_price=close_now,
        stop=sl_stop_now,
        delta_format=delta_format_now,
        hit_below=True
    )
    price_diff = abs(close_now - stop_price)
    size[c.i, c.col] = risk_amount_now * free_cash_now / price_diff

def generate_portfolio():
    ema_macd_strat = strat.run(
        high, 
        low,
        close,
        param_product=True,
            execute_kwargs=dict(
            engine='dask', 
            distribute="chunks", 
            chunk_len='auto'  
        )
    )

    #Shift arrays
    long_entries = (~ema_macd_strat.long_entry.isnull()).vbt.signals.fshift()  
    short_entries = (~ema_macd_strat.short_entry.isnull()).vbt.signals.fshift()
    sl_stop = shift_array(ema_macd_strat.SL_percentage)
    tp_stop = shift_array(ema_macd_strat.TP_percentage)
    #sl_stop = (~ema_macd_strat.SL_percentage.isnull()).vbt.fshift()
    #tp_stop = (~ema_macd_strat.TP_percentage.isnull()).vbt.fshift()

    pf1 = vbt.PF.from_signals(
        close, #close['BTCUSDT'].values,
        entries = ema_macd_strat.long_entry,
        short_entries = ema_macd_strat.short_entry,
        sl_stop = ema_macd_strat.SL_percentage,
        tp_stop = ema_macd_strat.TP_percentage,
        adjust_func_nb=adjust_func_nb,
        adjust_args=(
            vbt.Rep("size"), 
            vbt.Rep("sl_stop"), 
            vbt.Rep("delta_format"), 
            vbt.Rep("risk_amount")
        ),
        size=vbt.RepFunc(lambda wrapper: np.full(wrapper.shape_2d, np.nan)),
        delta_format="percent",
        init_cash=1000,
        leverage=3,
        leverage_mode=vbt.pf_enums.LeverageMode.Eager,
        broadcast_named_args=dict(risk_amount=0.01),
        freq=timeframe
    )
    return pf1

def refresh_strategy_html():
    pf1 = generate_portfolio()
    pf1.plot().write_html("backtest_chart.html")
    print("[INFO] Strategy HTML updated.")
    
if __name__ == "__main__":
    refresh_strategy_html()