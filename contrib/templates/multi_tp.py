"""Example:

```python
data = vbt.YFData.fetch("BTC-USD")
fast_sma = data.run("talib_sma", 20, hide_params=True)
slow_sma = data.run("talib_sma", 50, hide_params=True)
entries = fast_sma.real_crossed_above(slow_sma)
exits = fast_sma.real_crossed_below(slow_sma)

pf = multi_tp_from_signals(
    data,
    tp_stops=[0.25, 0.5, 0.75],
    tp_pcts=[0.5, 0.5, 1.0],
    long_entries=entries,
    long_exits=exits
)
```
"""

import vectorbtpro as vbt
import numpy as np
from numba import njit
from collections import namedtuple

MyContext = namedtuple(
    "MyContext",
    [
        "long_entries",
        "long_exits",
        "short_entries",
        "short_exits",
        "user_size",
        "user_size_type",
        "size",
        "size_type",
        "accumulate",
        "tp_stops",
        "tp_pcts",
        "temp_active_tp",
        "temp_init_idx",
        "temp_init_price",
    ],
)


@njit
def signal_func_nb(c, my_c):
    position_now = c.last_position[c.col]
    tp_info = c.last_tp_info[c.col]
    last_order = c.order_records[c.order_counts[c.col] - 1, c.col]

    is_long_entry = vbt.pf_nb.select_nb(c, my_c.long_entries)
    is_long_exit = vbt.pf_nb.select_nb(c, my_c.long_exits)
    is_short_entry = vbt.pf_nb.select_nb(c, my_c.short_entries)
    is_short_exit = vbt.pf_nb.select_nb(c, my_c.short_exits)
    user_size = vbt.pf_nb.select_nb(c, my_c.user_size)
    user_size_type = vbt.pf_nb.select_nb(c, my_c.user_size_type)

    if position_now == 0:
        if is_long_entry or is_short_entry:
            my_c.size[0] = user_size
            my_c.size_type[0] = user_size_type
            my_c.accumulate[0] = False
            if is_long_entry:
                return True, False, False, False
            if is_short_entry:
                return False, False, True, False

    if position_now > 0:
        if is_long_exit or is_short_entry:
            my_c.size[0] = user_size
            my_c.size_type[0] = user_size_type
            my_c.accumulate[0] = False
            if is_long_exit:
                return False, True, False, False
            if is_short_entry:
                return False, False, True, False

    if position_now < 0:
        if is_short_exit or is_long_entry:
            my_c.size[0] = user_size
            my_c.size_type[0] = user_size_type
            my_c.accumulate[0] = False
            if is_short_exit:
                return False, False, False, True
            if is_long_entry:
                return True, False, False, False

    def init_temp_info():
        my_c.temp_active_tp[0] = 0
        my_c.temp_init_idx[0] = tp_info.init_idx
        my_c.temp_init_price[0] = tp_info.init_price

    def update_temp_info():
        my_c.temp_active_tp[0] += 1

    def update_tp_stop():
        vbt.pf_nb.set_tp_info_nb(
            tp_info,
            init_idx=my_c.temp_init_idx[0],
            init_price=my_c.temp_init_price[0],
            stop=my_c.tp_stops[my_c.temp_active_tp[0]],
            exit_type=vbt.pf_enums.StopExitType.CloseReduce,
        )
        my_c.size[0] = my_c.tp_pcts[my_c.temp_active_tp[0]] * abs(position_now)
        my_c.size_type[0] = vbt.pf_enums.SizeType.Amount
        my_c.accumulate[0] = True

    if last_order.idx == c.i - 1:
        if last_order.stop_type == -1:
            init_temp_info()
            update_tp_stop()
        elif last_order.stop_type == vbt.sig_enums.StopType.TP:
            if my_c.temp_active_tp[0] < len(my_c.tp_pcts) - 1:
                update_temp_info()
                update_tp_stop()

    return False, False, False, False


def multi_tp_from_signals(
    data,
    *,
    tp_stops,
    tp_pcts,
    long_entries=False,
    long_exits=False,
    short_entries=False,
    short_exits=False,
    size=np.inf,
    size_type=vbt.pf_enums.SizeType.Amount,
    **kwargs
):
    my_c = vbt.RepEval(
        """MyContext(
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            user_size=user_size,
            user_size_type=user_size_type,
            size=size,
            size_type=size_type,
            accumulate=accumulate,
            tp_stops=vbt.to_1d_array(tp_stops),
            tp_pcts=vbt.to_1d_array(tp_pcts),
            temp_active_tp=np.full(1, -1),
            temp_init_idx=np.full(1, -1),
            temp_init_price=np.full(1, np.nan)
        )""",
        context=dict(MyContext=MyContext, tp_stops=tp_stops, tp_pcts=tp_pcts),
    )
    return vbt.Portfolio.from_signals(
        data,
        signal_func_nb=signal_func_nb,
        signal_args=(my_c,),
        size=np.full(1, np.nan),
        size_type=np.full(1, -1),
        accumulate=np.full(1, False),
        broadcast_named_args=dict(
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            user_size=size,
            user_size_type=size_type,
        ),
        **kwargs,
    )
