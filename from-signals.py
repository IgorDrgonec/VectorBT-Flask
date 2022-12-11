# %%
# Save this file by also removing the ".txt" extension
# Run either with Jupytext: https://jupytext.readthedocs.io/en/latest/install.html
# Or as a Python script; make sure to place print statements where appropriate

# %% [markdown]
# #  From signals

# %%
import numpy as np
import pandas as pd
from itertools import product

price = np.array([21, 20, 22, 21])
returns = (price[1:] - price[:-1]) / price[:-1]
permutations = list(product([False, True], repeat=len(returns)))
total_return = np.prod(1 + np.where(permutations, returns, -returns), axis=1) - 1
pd.Series(total_return, index=permutations).sort_values(ascending=False)

# %% [markdown]
# ## Mechanics
# ### Framework
# #### Segment workflow

# %% [markdown]
# ### Signal generation
# #### Signal function

# %%
import numpy as np
import pandas as pd
from numba import njit
import vectorbtpro as vbt

@njit
def signal_func_nb(c):
    return False, False, False, False

close = pd.DataFrame({
    "BTC-USD": [20594.29, 20719.41, 19986.60, 21084.64],
    "ETH-USD": [1127.51, 1125.37, 1051.32, 1143.20],
    "DOT-USD": [7.88, 7.74, 7.41, 7.78],
    "BNB-USD": [216.90, 219.67, 214.23, 228.92]
})

pf = vbt.Portfolio.from_signals(
    close=close,
    signal_func_nb=signal_func_nb
)
pf.order_records

# %%
@njit
def signal_func_nb(c):
    print(c.col, c.i)
    return False, False, False, False

pf = vbt.Portfolio.from_signals(
    close=close[["BTC-USD", "ETH-USD"]],
    signal_func_nb=signal_func_nb
)

# %%
@njit
def signal_func_nb(c):
    print(c.group, c.col, c.i)
    return False, False, False, False

pf = vbt.Portfolio.from_signals(
    close=close,
    signal_func_nb=signal_func_nb,
    group_by=[0, 0, 1, 1],
    cash_sharing=True
)

# %% [markdown]
# ### Signal resolution

# %%
vbt.pf_nb.resolve_signal_conflict_nb(
    position_now=20,
    is_entry=True,
    is_exit=True,
    direction=vbt.pf_enums.Direction.LongOnly,
    conflict_mode=vbt.pf_enums.ConflictMode.Adjacent
)

# %%
vbt.pf_nb.resolve_dir_conflict_nb(
    position_now=20,
    is_long_entry=True,
    is_short_entry=True,
    upon_dir_conflict=vbt.pf_enums.DirectionConflictMode.Short,
)

# %%
vbt.pf_nb.resolve_opposite_entry_nb(
    position_now=20,
    is_long_entry=False,
    is_long_exit=False,
    is_short_entry=True,
    is_short_exit=False,
    upon_opposite_entry=vbt.pf_enums.OppositeEntryMode.Close,
    accumulate=False,
)

# %% [markdown]
# ### Signal conversion

# %%
vbt.pf_nb.signal_to_size_nb(
    position_now=20,
    val_price_now=20594.29,
    value_now=411885.80,
    is_long_entry=False,
    is_long_exit=True,
    is_short_entry=False,
    is_short_exit=False,
    size=0.1,
    size_type=vbt.pf_enums.SizeType.ValuePercent,
    accumulate=False
)

# %%
vbt.pf_nb.signal_to_size_nb(
    position_now=20,
    val_price_now=20594.29,
    value_now=411885.80,
    is_long_entry=False,
    is_long_exit=False,
    is_short_entry=True,
    is_short_exit=False,
    size=0.1,
    size_type=vbt.pf_enums.SizeType.ValuePercent,
    accumulate=False
)

# %% [markdown]
# ### Main order resolution

# %% [markdown]
# ### Limit management
# #### Creation
# #### Expiration

# %%
vbt.pf_nb.check_limit_expired_nb(
    creation_idx=0,
    i=1,
    tif=pd.Timedelta("36h").to_timedelta64().astype(np.int64),
    index=pd.date_range("2020-01-01", periods=3).values.astype(np.int64),
    freq=pd.Timedelta("1d").to_timedelta64().astype(np.int64)
)

# %%
vbt.pf_nb.check_limit_expired_nb(
    creation_idx=0,
    i=1,
    tif=pd.Timedelta("24h").to_timedelta64().astype(np.int64),
    index=pd.date_range("2020-01-01", periods=3).values.astype(np.int64),
    freq=pd.Timedelta("1d").to_timedelta64().astype(np.int64)
)

# %% [markdown]
# #### Activation

# %%
vbt.pf_nb.check_limit_hit_nb(
    open=10.0,
    high=11.0,
    low=9.0,
    close=10.5,
    price=9.5,
    size=2.0
)

# %%
vbt.pf_nb.check_limit_hit_nb(
    open=10.0,
    high=11.0,
    low=9.0,
    close=10.5,
    price=11.0,
    size=2.0
)

# %% [markdown]
# #### Cancellation

# %%
vbt.pf_nb.resolve_pending_conflict_nb(
    is_pending_long=True,
    is_user_long=False,
    upon_adj_conflict=vbt.pf_enums.PendingConflictMode.KeepIgnore,
    upon_opp_conflict=vbt.pf_enums.PendingConflictMode.CancelIgnore,
)

# %% [markdown]
# ### Stop management
# #### Types
# #### Creation
# #### Activation

# %%
vbt.pf_nb.check_stop_hit_nb(
    open=10.0,
    high=11.0,
    low=9.0,
    close=10.5,
    is_position_long=True,
    stop_price=10.0,
    stop=0.1
)

# %%
vbt.pf_nb.check_stop_hit_nb(
    open=10.0,
    high=11.0,
    low=9.0,
    close=10.5,
    is_position_long=True,
    stop_price=12.0,
    stop=0.1
)

# %% [markdown]
# #### Resolution

# %%
vbt.pf_nb.generate_stop_signal_nb(
    position_now=20,
    stop_exit_type=vbt.pf_enums.StopExitType.Reverse
)

# %%
vbt.pf_nb.resolve_stop_exit_price_nb(
    stop_price=9.0,
    close=10.5,
    stop_exit_price=vbt.pf_enums.StopExitPrice.Stop
)

# %%
vbt.pf_nb.resolve_stop_exit_price_nb(
    stop_price=9.0,
    close=10.5,
    stop_exit_price=9.5
)

# %% [markdown]
# #### Updating

# %%
vbt.pf_nb.should_update_stop_nb(
    new_stop=0.1,
    upon_stop_update=vbt.pf_enums.StopUpdateMode.Override
)

# %%
vbt.pf_nb.should_update_stop_nb(
    new_stop=np.nan,
    upon_stop_update=vbt.pf_enums.StopUpdateMode.Override
)

# %%
vbt.pf_nb.should_update_stop_nb(
    new_stop=np.nan,
    upon_stop_update=vbt.pf_enums.StopUpdateMode.OverrideNaN
)

# %% [markdown]
# #### Cancellation
# ## Signals

# %%
data = vbt.BinanceData.fetch(["BTCUSDT", "ETHUSDT"])

# %%
sub_data = data.loc["2021-02-18":"2021-02-24"]

vbt.settings.set_theme("dark")
sub_data.plot(symbol="BTCUSDT").show()

# %%
pf = vbt.Portfolio.from_signals(sub_data)
pf.orders.count()

# %%
X = True
O = False

entries = pd.Series([X, O, O, O, O, O, O])
exits   = pd.Series([O, O, O, X, O, O, O])
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=entries,
    exits=exits
)
pf.orders.records_readable

# %%
pf.assets

# %%
entries = pd.Series([X, O, O, O, O, O, O])
exits = pd.DataFrame({
    0: [O, O, O, X, O, O, O],
    1: [O, O, X, O, O, O, O],
})
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=entries,
    exits=exits
)
pf.assets

# %%
exits = sub_data.symbol_wrapper.fill(False)
exits.loc["2021-02-21", "BTCUSDT"] = True
exits.loc["2021-02-20", "ETHUSDT"] = True
exits

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=vbt.index_dict({0: True}),
    exits=vbt.index_dict({
        vbt.ElemIdx("2021-02-21", "BTCUSDT"): True,
        vbt.ElemIdx("2021-02-20", "ETHUSDT"): True
    })
)
pf.assets

# %% [markdown]
# ### Direction-unaware

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=entries,
    exits=exits,
    direction="both"
)
pf.assets

# %%
direction = pd.DataFrame([["longonly", "shortonly"]])
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=entries,
    exits=exits,
    direction=direction
)
pf.assets

# %%
L = vbt.pf_enums.Direction.LongOnly
S = vbt.pf_enums.Direction.ShortOnly

pf = vbt.Portfolio.from_signals(
    sub_data.select("BTCUSDT"),
    entries=  pd.Series([X, O, O, O, X, O, O]),
    exits=    pd.Series([O, O, O, X, O, O, X]),
    direction=pd.Series([L, L, L, L, S, S, S])
)
pf.assets

# %% [markdown]
# ### Direction-aware

# %%
pf = vbt.Portfolio.from_signals(
    sub_data.select("BTCUSDT"),
    entries=      pd.Series([X, O, O, O, O, O, O]),
    exits=        pd.Series([O, O, O, X, O, O, O]),
    short_entries=pd.Series([O, O, O, O, X, O, O]),
    short_exits=  pd.Series([O, O, O, O, O, O, X]),
)
pf.assets

# %% [markdown]
# ### Signal function

# %%
@njit
def signal_func_nb(c):
    ts = c.index[c.i]
    if vbt.dt_nb.matches_date_nb(ts, 2021, 2, 18):
        return True, False, False, False
    if vbt.dt_nb.matches_date_nb(ts, 2021, 2, 21):
        return False, True, False, False
    if vbt.dt_nb.matches_date_nb(ts, 2021, 2, 22):
        return False, False, True, False
    if vbt.dt_nb.matches_date_nb(ts, 2021, 2, 24):
        return False, False, False, True
    return False, False, False, False

pf = vbt.Portfolio.from_signals(
    sub_data.select("BTCUSDT"),
    signal_func_nb=signal_func_nb
)
pf.assets

# %%
@njit
def signal_func_nb(c, entries, exits, short_entries, short_exits):
    long_entry = entries[c.i]
    long_exit = exits[c.i]
    short_entry = short_entries[c.i]
    short_exit = short_exits[c.i]
    return long_entry, long_exit, short_entry, short_exit

pf = vbt.Portfolio.from_signals(
    sub_data.select("BTCUSDT"),
    signal_func_nb=signal_func_nb,
    signal_args=(
        pd.Series([X, O, O, O, O, O, O]).values,
        pd.Series([O, O, O, X, O, O, O]).values,
        pd.Series([O, O, O, O, X, O, O]).values,
        pd.Series([O, O, O, O, O, O, X]).values
    )
)
pf.assets

# %%
@njit
def signal_func_nb(c, entries, exits, short_entries, short_exits):
    long_entry = vbt.pf_nb.select_nb(c, entries)
    long_exit = vbt.pf_nb.select_nb(c, exits)
    short_entry = vbt.pf_nb.select_nb(c, short_entries)
    short_exit = vbt.pf_nb.select_nb(c, short_exits)
    return long_entry, long_exit, short_entry, short_exit

pf = vbt.Portfolio.from_signals(
    sub_data,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.to_2d_array(pd.Series([X, O, O, O, O, O, O])),
        vbt.to_2d_array(pd.Series([O, O, O, X, O, O, O])),
        vbt.to_2d_array(pd.Series([O, O, O, O, X, O, O])),
        vbt.to_2d_array(pd.Series([O, O, O, O, O, O, X]))
    )
)
pf.assets

# %%
entries = pd.Series({pd.Timestamp("2021-02-18 00:00:00+00:00"): True})
exits = pd.Series({pd.Timestamp("2021-02-21 00:00:00+00:00"): True})
short_entries = pd.Series({pd.Timestamp("2021-02-22 00:00:00+00:00"): True})
short_exits = pd.Series({pd.Timestamp("2021-02-24 00:00:00+00:00"): True})

pf = vbt.Portfolio.from_signals(
    sub_data,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entries"),
        vbt.Rep("exits"),
        vbt.Rep("short_entries"),
        vbt.Rep("short_exits")
    ),
    broadcast_named_args=dict(
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits
    )
)
pf.assets

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("entries"),
        vbt.Rep("exits"),
        vbt.Rep("short_entries"),
        vbt.Rep("short_exits")
    ),
    broadcast_named_args=dict(
        entries=      vbt.index_dict({"2021-02-18": True, "_def": False}),
        exits=        vbt.index_dict({"2021-02-21": True, "_def": False}),
        short_entries=vbt.index_dict({"2021-02-22": True, "_def": False}),
        short_exits=  vbt.index_dict({"2021-02-24": True, "_def": False})
    )
)

# %%
>>> @njit
... def signal_func_nb(c, fast_sma, slow_sma, wait):
...     curr_wait = vbt.pf_nb.select_nb(c, wait)  # (1)!
...     i_wait = c.i - curr_wait  # (2)!
...     if i_wait < 0:  # (3)!
...         return False, False, False, False
...
...     if vbt.nb.iter_crossed_above_nb(fast_sma, slow_sma, i_wait, c.col):  # (4)!
...         cross_confirmed = True
...         for j in range(i_wait + 1, c.i + 1):  # (5)!
...             if not vbt.nb.iter_above_nb(fast_sma, slow_sma, j, c.col):
...                 cross_confirmed = False
...                 break
...         if cross_confirmed:
...             return True, False, False, False
...
...     if vbt.nb.iter_crossed_below_nb(fast_sma, slow_sma, i_wait, c.col):  # (6)!
...         cross_confirmed = True
...         for j in range(i_wait + 1, c.i + 1):
...             if not vbt.nb.iter_below_nb(fast_sma, slow_sma, j, c.col):
...                 cross_confirmed = False
...                 break
...         if cross_confirmed:
...             return False, False, True, False
...
...     return False, False, False, False

>>> fast_sma = data.run("sma", 20, short_name="fast_sma").real
>>> slow_sma = data.run("sma", 50, short_name="slow_sma").real
>>> pf = vbt.Portfolio.from_signals(
...     data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.Rep("fast_sma"),
...         vbt.Rep("slow_sma"),
...         vbt.Rep("wait")
...     ),
...     broadcast_named_args=dict(
...         fast_sma=fast_sma,
...         slow_sma=slow_sma,
...         wait=0  # (7)!
...     )
... )
>>> pf.orders.count()

# %%
n_crossed_above = fast_sma.vbt.crossed_above(slow_sma).sum()
n_crossed_below = fast_sma.vbt.crossed_below(slow_sma).sum()
n_crossed_above + n_crossed_below

# %%
pf = vbt.Portfolio.from_signals(
    data,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("fast_sma"),
        vbt.Rep("slow_sma"),
        vbt.Rep("wait")
    ),
    broadcast_named_args=dict(
        fast_sma=fast_sma,
        slow_sma=slow_sma,
        wait=vbt.Param([0, 1, 7, 30])
    )
)
pf.orders.count()

# %%
@njit
def signal_func_nb(c, fast_sma, slow_sma, wait, temp_coi):
    if temp_coi[c.col] != -1:
        crossed_above = vbt.nb.iter_crossed_above_nb(
            fast_sma, slow_sma, temp_coi[c.col], c.col
        )
        crossed_below = vbt.nb.iter_crossed_below_nb(
            fast_sma, slow_sma, temp_coi[c.col], c.col
        )
        if crossed_above:
            if not vbt.pf_nb.iter_above_nb(c, fast_sma, slow_sma):
                temp_coi[c.col] = -1
        if crossed_below:
            if not vbt.pf_nb.iter_below_nb(c, fast_sma, slow_sma):
                temp_coi[c.col] = -1

    curr_wait = vbt.pf_nb.select_nb(c, wait)
    if temp_coi[c.col] != -1:
        if c.i - temp_coi[c.col] == curr_wait:
            if crossed_above:
                temp_coi[c.col] = -1
                return True, False, False, False
            if crossed_below:
                temp_coi[c.col] = -1
                return False, False, True, False
    else:
        if vbt.pf_nb.iter_crossed_above_nb(c, fast_sma, slow_sma):
            if curr_wait == 0:
                return True, False, False, False
            temp_coi[c.col] = c.i
        if vbt.pf_nb.iter_crossed_below_nb(c, fast_sma, slow_sma):
            if curr_wait == 0:
                return False, False, True, False
            temp_coi[c.col] = c.i

    return False, False, False, False

pf = vbt.Portfolio.from_signals(
    data,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("fast_sma"),
        vbt.Rep("slow_sma"),
        vbt.Rep("wait"),
        vbt.RepEval("np.full(wrapper.shape_2d[1], -1)")
    ),
    broadcast_named_args=dict(
        fast_sma=fast_sma,
        slow_sma=slow_sma,
        wait=vbt.Param([0, 1, 7, 30])
    )
)
pf.orders.count()

# %% [markdown]
# ### Conflicts

# %%
pf = vbt.Portfolio.from_signals(sub_data, entries=True)
pf.asset_flow

# %%
pf = vbt.Portfolio.from_signals(sub_data, entries=True, exits=True)
pf.asset_flow

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=True,
    exits=True,
    upon_long_conflict="entry"
)
pf.asset_flow

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=True,
    exits=True,
    direction="both",
    upon_dir_conflict="short"
)
pf.asset_flow

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=True,
    exits=vbt.index_dict({0: False, "_def": True}),
    short_entries=vbt.index_dict({0: False, "_def": True}),
    short_exits=vbt.index_dict({0: False, "_def": True}),
    upon_long_conflict="entry",
    upon_short_conflict="entry",
    upon_dir_conflict="opposite",
    upon_opposite_entry="reverse"
)
pf.asset_flow

# %% [markdown]
# ## Orders

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    exits=  pd.Series([O, O, O, X, O, O, O]),
    size=1,
    size_type="value"
)
pf.assets

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=      pd.Series([X,   O, O, O, O,    O, O]),
    exits=        pd.Series([O,   O, O, X, O,    O, O]),
    short_entries=pd.Series([O,   O, O, O, X,    O, O]),
    short_exits=  pd.Series([O,   O, O, O, O,    O, X]),
    size=         pd.Series([0.5, 0, 0, 0, 0.25, 0, 0]),
    size_type="valuepercent",
)
pf.asset_value / pf.value

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    exits=  pd.Series([O, O, O, X, O, O, O]),
    size=     vbt.Param([np.inf,   1,       0.5], level=0),
    size_type=vbt.Param(["amount", "value", "valuepercent"], level=0)
)
pf.total_return

# %% [markdown]
# ### Accumulation

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    exits=  pd.Series([O, O, O, X, O, O, O]),
    size=1,
    size_type="value",
    accumulate=True
)
pf.assets

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=True,
    size=1,
    size_type="value",
    accumulate=vbt.Param([False, True])
)
pf.asset_flow

# %% [markdown]
# ### Size types
# ### Size granularity

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    exits=  pd.Series([O, O, O, X, O, O, O]),
    size_granularity=vbt.Param([1, 0.001]),
    init_cash=1000_000
)
pf.asset_flow

# %% [markdown]
# ### Price

# %%
price = sub_data.symbol_wrapper.fill()
entries = pd.Series([X, O, O, O, O, O, O]).vbt.broadcast_to(price)
exits =   pd.Series([O, O, O, X, O, O, O]).vbt.broadcast_to(price)
price[entries] = sub_data.open
price[exits] = sub_data.close

pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=entries,
    exits=exits,
    price=price
)

pf.orders.price.to_pd() == sub_data.open

# %%
pf.orders.price.to_pd() == sub_data.close

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    exits=  pd.Series([O, O, O, X, O, O, O]),
    price="nextopen"
)
pf.orders.price.to_pd() == sub_data.open

# %% [markdown]
# ### Shifting

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]).vbt.signals.fshift(),
    exits=  pd.Series([O, O, O, X, O, O, O]).vbt.signals.fshift(),
    price="open"
)
pf.orders.price.to_pd() == sub_data.open

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=  pd.Series([X, O, O, O, X, O, O]).vbt.fshift(1, False),
    exits=    pd.Series([O, O, O, X, O, O, X]).vbt.fshift(1, False),
    direction=pd.Series([L, L, L, L, S, S, S]).vbt.fshift(1, -1)
)
pf.assets

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=  pd.Series([X, O, O, O, X, O, O]),
    exits=    pd.Series([O, O, O, X, O, O, X]),
    direction=pd.Series([L, L, L, L, S, S, S]),
    from_ago=1
)
pf.assets

# %% [markdown]
# ### Slippage

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    exits=  pd.Series([O, O, O, X, O, O, O]),
    price="nextopen",
    slippage=0.005
)
pf.orders.price.to_pd()

# %%
pf.orders.price.to_pd() / sub_data.open

# %% [markdown]
# ### Limit orders

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    exits=  pd.Series([O, O, O, X, O, O, O]),
    order_type="limit"
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([O, O, O, O, X, O, O]),
    direction="shortonly",
    price=sub_data.high.vbt.fshift(),
    order_type="limit"
)
pf.orders.price.to_pd()

# %% [markdown]
# #### Time in force

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([O, O, X, O, O, O, O]),
    price=sub_data.low.vbt.fshift(),
    order_type="limit"
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([O, O, X, O, O, O, O]),
    price=sub_data.low.vbt.fshift(),
    order_type="limit",
    limit_tif="2d"
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([O, O, X, O, O, O, O]),
    price=sub_data.low.vbt.fshift(),
    order_type="limit",
    limit_tif=vbt.Param([
        -1,
        pd.Timedelta(days=1),
        2 * np.timedelta64(86400000000000),
        3 * vbt.dt_nb.d_ns
    ], keys=["none", "1 days", "2 days", "3 days"])
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([O, O, X, O, O, O, O]),
    price=sub_data.low.vbt.fshift(),
    order_type="limit",
    limit_tif=vbt.Param(
        [-1, 1, 2, 3],
        keys=["none", "1 rows", "2 rows", "3 rows"]
    ),
    time_delta_format="rows"
)

# %%
index_td = vbt.dt.freq_to_timedelta("1 minute")
tif_td = vbt.dt.freq_to_timedelta("1 day")
int(tif_td / index_td)

# %% [markdown]
# #### Expiration date

# %%
sub_data.symbol_wrapper.get_period_ns_index("1d")

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([O, O, X, O, O, O, O]),
    price=sub_data.low.vbt.fshift(),
    order_type="limit",
    limit_expiry="W-MON"
)
pf.orders.price.to_pd()

# %%
sub_data.symbol_wrapper.index.to_period("W-MON")

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([O, O, X, O, O, O, O]),
    price=sub_data.low.vbt.fshift(),
    order_type="limit",
    limit_expiry="W-SUN"
)
pf.orders.price.to_pd()

# %%
sub_data.symbol_wrapper.index.to_period("W-SUN")

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([O, O, X, O, O, O, O]),
    price=sub_data.low.vbt.fshift(),
    order_type="limit",
    limit_expiry=vbt.RepEval("""
    expiry_index = wrapper.index + pd.Timedelta(days=2)
    expiry_arr = vbt.to_2d_array(expiry_index.values.astype(np.int64))
    expiry_arr
    """)
)
pf.orders.price.to_pd()

# %% [markdown]
# #### Conflicts

# %%
all_conflict_modes = vbt.pf_enums.PendingConflictMode._fields
pf = vbt.Portfolio.from_signals(
    sub_data.select("BTCUSDT"),
    entries=      pd.Series([O, O, X, O, O, O, O]),
    short_entries=pd.Series([O, O, O, X, O, O, O]),
    price=sub_data.select("BTCUSDT").low.vbt.fshift(),
    order_type=vbt.index_dict({2: "limit", "_def": "market"}),
    upon_opp_limit_conflict=vbt.Param(all_conflict_modes)
)
pf.orders.price.to_pd()

# %% [markdown]
# #### Delta

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    order_type="limit",
    limit_delta=vbt.Param([0, 0.1, 0.5])
)
pf.orders.price.to_pd()

# %% [markdown]
# #### Reversing

# %%
pf = vbt.Portfolio.from_signals(
    sub_data.select("BTCUSDT"),
    entries=pd.Series([X, O, O, O, O, O, O]),
    order_type="limit",
    limit_delta=vbt.Param([0, 100, 5000, 10000]),
    delta_format="absolute",
    limit_reverse=True
)
pf.orders.price.to_pd()

# %% [markdown]
# #### Adjustment

# %%
@njit
def adjust_func_nb(c):
    limit_info = c.last_limit_info[c.col]
    if limit_info.creation_idx != -1:
        if c.i - limit_info.creation_idx >= 1:
            vbt.pf_nb.clear_limit_info_nb(limit_info)

pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    price="open",
    order_type="limit",
    limit_delta=vbt.Param([0, 0.1]),
    adjust_func_nb=adjust_func_nb
)
pf.orders.price.to_pd()

# %%
@njit
def adjust_func_nb(c):
    limit_info = c.last_limit_info[c.col]
    if limit_info.creation_idx != -1:
        if c.i - limit_info.creation_idx >= 1:
            limit_info.init_price = -np.inf
            limit_info.delta = 0.01

pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    price="open",
    order_type="limit",
    limit_delta=vbt.Param([0.1, 0.2]),
    adjust_func_nb=adjust_func_nb
)
pf.orders.price.to_pd()

# %%
@njit
def adjust_func_nb(c, custom_delta):
    limit_info = c.last_limit_info[c.col]
    if c.i == 0:
        curr_delta = vbt.pf_nb.select_nb(c, custom_delta)
        vbt.pf_nb.set_limit_info_nb(limit_info, c.i, delta=curr_delta)

pf = vbt.Portfolio.from_signals(
    sub_data,
    broadcast_named_args=dict(custom_delta=vbt.Param([0, 0.1])),
    adjust_func_nb=adjust_func_nb,
    adjust_args=(vbt.Rep("custom_delta"),)
)
pf.orders.price.to_pd()

# %% [markdown]
# ### Stop orders

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop=0.1,
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop=vbt.Param([np.nan, 0.1, 0.2]),
)
pf.orders.price.to_pd()

# %%
atr = data.run("atr").atr
sub_atr = atr.loc["2021-02-18":"2021-02-24"]

pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop=sub_atr / sub_data.close
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data.select("BTCUSDT"),
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop=100,
    delta_format="absolute"
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop=1 * sub_atr / sub_data.close,
    tp_stop=2 * sub_atr / sub_data.close
)
pf.orders.price.to_pd()

# %%
pf.orders.records_readable

# %%
pf.orders.stop_type.to_pd(mapping=True)

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop= vbt.Param([0.1,    np.nan, np.nan, np.nan], level=0),
    tsl_stop=vbt.Param([np.nan, 0.1,    0.1,    np.nan], level=0),
    tsl_th=  vbt.Param([np.nan, np.nan, 0.1,    np.nan], level=0),
    tp_stop= vbt.Param([np.nan, np.nan, np.nan, 0.1   ], level=0),
)
pf.total_return

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop=vbt.Param([np.nan, 0.1]),
    tsl_stop=vbt.Param([np.nan, 0.1]),
    tsl_th=vbt.Param([np.nan, 0.1]),
    tp_stop=vbt.Param([np.nan, 0.1]),
)

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop=vbt.Param([np.nan, 0.1], level=0),
    tsl_stop=vbt.Param([np.nan, 0.1, 0.1], level=1),
    tsl_th=vbt.Param([np.nan, np.nan, 0.1], level=1),
    tp_stop=vbt.Param([np.nan, 0.1], level=2),
)

# %%
total_return = pf.total_return
sl_stops = total_return.index.get_level_values("sl_stop")
tp_stops = total_return.index.get_level_values("tp_stop")

total_return[np.isnan(sl_stops) | np.isnan(tp_stops)].mean()

# %%
total_return[~np.isnan(sl_stops) & ~np.isnan(tp_stops)].mean()

# %%
StopOrderPrep = vbt.IF.from_expr("""
    sl_stop = @p_sl_mult * @in_atr / @in_close
    tp_stop = @p_tp_mult * @in_atr / @in_close
    sl_stop, tp_stop
""")
stop_order_prep = StopOrderPrep.run(
    close=sub_data.close,
    atr=sub_atr,
    sl_mult=[np.nan, 1],
    tp_mult=[np.nan, 1],
    param_product=True
)
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    sl_stop=stop_order_prep.sl_stop,
    tp_stop=stop_order_prep.tp_stop,
)
pf.total_return

# %% [markdown]
# #### Entry point

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    price="open",
    tp_stop=0.005,
    stop_entry_price="fillprice"
)
pf.orders.price.to_pd()

# %% [markdown]
# #### Exit point

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    tp_stop=0.05,
    stop_exit_price=vbt.Param(["stop", "close"])
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    tp_stop=0.01,
    stop_exit_type=vbt.Param(["close", "reverse"])
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    tp_stop=0.01,
    stop_exit_type=vbt.index_dict({0: "reverse", "_def": "close"})
)
pf.orders.price.to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    tp_stop=0.05,
    stop_order_type=vbt.Param(["market", "limit"]),
    stop_limit_delta=0.01
)
pf.orders.price.to_pd()

# %% [markdown]
# #### Conflicts

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, X, O, X, O, X]),
    exits=  pd.Series([O, X, O, X, O, X, O]),
    direction="both",
    sl_stop=0.1,
    upon_adj_stop_conflict="keepignore",
    upon_opp_stop_conflict="keepignore",
)
pf.orders.price.to_pd()

# %% [markdown]
# #### Adjustment

# %%
@njit
def adjust_func_nb(c, max_loss, max_profit):
    position_now = c.last_position[c.col]
    if position_now != 0:
        sl_info = c.last_sl_info[c.col]
        tp_info = c.last_tp_info[c.col]
        ml = vbt.pf_nb.select_nb(c, max_loss)
        mp = vbt.pf_nb.select_nb(c, max_profit)

        if not vbt.pf_nb.is_stop_active_nb(sl_info):
            sl_info.stop = ml / (sl_info.init_price * position_now)
        if not vbt.pf_nb.is_stop_active_nb(tp_info):
            tp_info.stop = mp / (sl_info.init_price * position_now)

pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    adjust_func_nb=adjust_func_nb,
    adjust_args=(vbt.Rep("max_loss"), vbt.Rep("max_profit")),
    broadcast_named_args=dict(
        max_loss=10,
        max_profit=10
    )
)
pf.exit_trades.pnl.to_pd()

# %% [markdown]
# ### Dynamic

# %%
>>> @njit
... def signal_func_nb(c, long_signals, short_signals, size):
...     long_signal = vbt.pf_nb.select_nb(c, long_signals)
...     short_signal = vbt.pf_nb.select_nb(c, short_signals)
...     if long_signal:
...         size[:] = 10  # (1)!
...     if short_signal:
...         size[:] = 5
...     return long_signal, False, short_signal, False

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(
...         vbt.Rep("long_signals"), 
...         vbt.Rep("short_signals"), 
...         vbt.Rep("size")
...     ),
...     size=np.full((1, 1), np.nan),  # (2)!
...     size_type="value",
...     accumulate=True,
...     broadcast_named_args=dict(
...         long_signals= pd.Series([X, O, O, O, X, O, O]),
...         short_signals=pd.Series([O, O, X, O, O, O, X]),
...     )
... )
>>> pf.orders.value.to_pd()

# %%
memory = {}
pf = vbt.Portfolio.from_signals(
    # ...
    size=vbt.RepEval("""
        size = np.full(wrapper.shape, np.nan)
        memory["size"] = size
        return size
        """,
        context=dict(memory=memory),
        context_merge_kwargs=dict(nested=False)
    ),
    # ...
)
memory["size"]

# %%
@njit
def signal_func_nb(c, long_signals, short_signals, size):
    long_signal = vbt.pf_nb.select_nb(c, long_signals)
    short_signal = vbt.pf_nb.select_nb(c, short_signals)
    if long_signal:
        size[0] = 10
    if short_signal:
        size[0] = 5
    return long_signal, False, short_signal, False

pf = vbt.Portfolio.from_signals(
    sub_data,
    signal_func_nb=signal_func_nb,
    signal_args=(
        vbt.Rep("long_signals"),
        vbt.Rep("short_signals"),
        vbt.Rep("size")
    ),
    size=np.full(1, np.nan),
    size_type="value",
    accumulate=True,
    broadcast_named_args=dict(
        long_signals= pd.Series([X, O, O, O, X, O, O]),
        short_signals=pd.Series([O, O, X, O, O, O, X]),
    )
)
pf.orders.value.to_pd()

# %% [markdown]
# #### TP ladder

# %%
>>> from collections import namedtuple

>>> Signals = namedtuple("Signals", [  # (1)!
...     "entries", 
...     "exits"
... ])
>>> Orders = namedtuple("Orders", [  # (2)!
...     "size", 
...     "accumulate"
... ])
>>> Ladders = namedtuple("Ladders", [  # (3)!
...     "tp1", 
...     "tp2", 
...     "tp1_pct", 
...     "tp2_pct"
... ])

>>> ladder_info_dt = np.dtype([  # (4)!
...     ("init_idx", np.int_),
...     ("init_price", np.float_),
...     ("tp1", np.float_),
...     ("tp2", np.float_),
...     ("tp1_pct", np.float_),
...     ("tp2_pct", np.float_)
... ], align=True)

>>> @njit  # (5)!
... def signal_func_nb(c, signals, orders, ladders, last_ladder_info):  # (6)!
...     is_entry = vbt.pf_nb.select_nb(c, signals.entries)  # (7)!
...     is_exit = vbt.pf_nb.select_nb(c, signals.exits)
...     position_now = c.last_position[c.col]
...     if position_now == 0 and is_entry:
...         orders.size[0, c.col] = np.inf  # (8)!
...         orders.accumulate[0, c.col] = False
...         return True, False, False, False
...     if position_now > 0 and is_exit:
...         orders.size[0, c.col] = np.inf
...         orders.accumulate[0, c.col] = False
...         return False, True, False, False
...     if position_now == 0:
...         return False, False, False, False
...
...     tp_info = c.last_tp_info[c.col]
...     ladder_info = last_ladder_info[c.col]  # (9)!
...     last_order = c.order_records[c.order_counts[c.col] - 1, c.col]  # (10)!
...     if not vbt.pf_nb.is_stop_active_nb(tp_info):  # (11)!
...         if last_order.stop_type == -1:  # (12)!
...             ladder_info.init_idx = tp_info.init_idx  # (13)!
...             ladder_info.init_price = tp_info.init_price
...             ladder_info.tp1 = vbt.pf_nb.select_nb(c, ladders.tp1)
...             ladder_info.tp2 = vbt.pf_nb.select_nb(c, ladders.tp2)
...             ladder_info.tp1_pct = vbt.pf_nb.select_nb(c, ladders.tp1_pct)
...             ladder_info.tp2_pct = vbt.pf_nb.select_nb(c, ladders.tp2_pct)
...             tp_info.stop = ladder_info.tp1  # (14)!
...             tp_info.exit_type = vbt.pf_enums.StopExitType.CloseReduce
...             orders.size[0, c.col] = ladder_info.tp1_pct * position_now
...             orders.accumulate[0, c.col] = True
...         else:  # (15)!
...             vbt.pf_nb.set_tp_info_nb(  # (16)!
...                 tp_info, 
...                 ladder_info.init_idx,  # (17)!
...                 init_price=ladder_info.init_price,
...                 stop=ladder_info.tp2,
...                 exit_type=vbt.pf_enums.StopExitType.CloseReduce
...             )
...             orders.size[0, c.col] = ladder_info.tp2_pct * position_now  # (18)!
...             orders.accumulate[0, c.col] = True
...     return False, False, False, False

>>> pf = vbt.Portfolio.from_signals(
...     sub_data,
...     signal_func_nb=signal_func_nb,
...     signal_args=(  # (19)!
...         vbt.RepEval(
...             "Signals(entries, exits)",
...             context=dict(Signals=Signals)  # (20)!
...         ),
...         vbt.RepEval(
...             "Orders(size, accumulate)",
...             context=dict(Orders=Orders)
...         ),
...         vbt.RepEval(
...             "Ladders(tp1, tp2, tp1_pct, tp2_pct)",
...             context=dict(Ladders=Ladders)
...         ),
...         vbt.RepEval(
...             "np.empty(wrapper.shape_2d[1], dtype=ladder_info_dt)", 
...             context=dict(ladder_info_dt=ladder_info_dt)
...         )
...     ),
...     size=vbt.RepEval("np.full((1, wrapper.shape_2d[1]), np.inf)"),
...     accumulate=vbt.RepEval("np.full((1, wrapper.shape_2d[1]), False)"),
...     broadcast_named_args=dict(
...         entries=pd.Series([X, O, O, O, O, O, O]), 
...         exits=  pd.Series([O, O, O, O, O, O, O]),
...         tp1=0.05,
...         tp2=0.1,
...         tp1_pct=0.5,
...         tp2_pct=1.0
...     )
... )
>>> pf.asset_flow

# %% [markdown]
# ## Grouping
# ### After simulation

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=      pd.Series([X, O, O, O, O, O, O]),
    exits=        pd.Series([O, O, O, X, O, O, O]),
    short_entries=pd.Series([O, O, O, O, X, O, O]),
    short_exits=  pd.Series([O, O, O, O, O, O, X]),
)
pf.get_value(group_by=True)

# %%
grouped_pf = pf.replace(wrapper=pf.wrapper.replace(group_by=True))
grouped_pf.value

# %%
grouped_pf.asset_flow

# %% [markdown]
# ### Before simulation

# %%
@njit
def signal_func_nb(c, entries, exits):
    is_entry = vbt.pf_nb.select_nb(c, entries)
    is_exit = vbt.pf_nb.select_nb(c, exits)
    other_in_position = False
    for col in range(c.from_col, c.to_col):
        if col != c.col and c.last_position[col] != 0:
            other_in_position = True
            break
    if other_in_position:
        return False, False, False, False
    return is_entry, is_exit, False, False

pf = vbt.Portfolio.from_signals(
    sub_data,
    signal_func_nb=signal_func_nb,
    signal_args=(vbt.Rep("entries"), vbt.Rep("exits")),
    broadcast_named_args=dict(
        entries=pd.DataFrame({
            0: [X, O, O, O, O, O, X],
            1: [O, X, O, X, O, O, O]
        }),
        exits=pd.DataFrame({
            0: [O, O, X, O, X, O, O],
            1: [O, O, O, O, O, X, O]
        })
    ),
    group_by=True
)
pf.asset_flow

# %% [markdown]
# #### Sorting

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    price=["close", "open"],
    size=[100, 50],
    size_type="value",
    group_by=True,
    cash_sharing=True
)
pf.orders.get_value(group_by=False).to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    price="close",
    size=[100, 50],
    size_type="value",
    group_by=True,
    cash_sharing=True
)
pf.orders.get_value(group_by=False).to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    price="close",
    size=[100, 50],
    size_type="value",
    group_by=True,
    cash_sharing=True,
    call_seq="auto"
)
pf.orders.get_value(group_by=False).to_pd()

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    price=["close", "open"],
    size=[100, 50],
    size_type="value",
    group_by=True,
    cash_sharing=True,
    call_seq="auto"
)

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=pd.Series([X, O, O, O, O, O, O]),
    price=sub_data.close,
    size=[100, 50],
    size_type="value",
    group_by=True,
    cash_sharing=True,
    call_seq="auto"
)

# %% [markdown]
# ## Custom outputs

# %%
pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=      pd.Series([X, O, O, O, O, O, O]),
    exits=        pd.Series([O, O, O, X, O, O, O]),
    short_entries=pd.Series([O, O, O, O, X, O, O]),
    short_exits=  pd.Series([O, O, O, O, O, O, X]),
    fill_returns=True
)
pf.returns

# %%
pf.get_returns()

# %%
pf.in_outputs

# %%
pf.get_in_output("returns")

# %%
@njit
def post_segment_func_nb(c):
    returns = c.in_outputs.returns
    total_return = c.in_outputs.total_return
    i = c.i
    g = c.group
    if c.cash_sharing:
        returns[i, g] = c.last_return[g]
        total_return[g] = c.last_value[g] / c.init_cash[g] - 1
    else:
        for col in range(c.from_col, c.to_col):
            returns[i, col] = c.last_return[col]
            total_return[col] = c.last_value[col] / c.init_cash[col] - 1

pf = vbt.Portfolio.from_signals(
    sub_data,
    entries=      pd.Series([X, O, O, O, O, O, O]),
    exits=        pd.Series([O, O, O, X, O, O, O]),
    short_entries=pd.Series([O, O, O, O, X, O, O]),
    short_exits=  pd.Series([O, O, O, O, O, O, X]),
    post_segment_func_nb=post_segment_func_nb,
    in_outputs=dict(
        returns=vbt.RepEval(
            "np.full((target_shape[0], len(cs_group_lens)), np.nan)"
        ),
        total_return=vbt.RepEval(
            "np.full(len(cs_group_lens), np.nan)"
        )
    )
)
pd.testing.assert_frame_equal(
    pf.get_in_output("returns"),
    pf.get_returns()
)
pd.testing.assert_series_equal(
    pf.get_in_output("total_return"),
    pf.get_total_return()
)

# %%
