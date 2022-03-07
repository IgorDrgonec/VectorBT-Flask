import os
import uuid
from collections import namedtuple
from copy import deepcopy
from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from numba import njit, typeof
from numba.typed import List

import vectorbtpro as vbt
from tests.utils import assert_records_close
from vectorbtpro.generic.enums import drawdown_dt
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.call_seq import build_call_seq, build_call_seq_nb
from vectorbtpro.portfolio.enums import *
from vectorbtpro.utils.random_ import set_seed

qs_available = True
try:
    import quantstats as qs
except:
    qs_available = False

seed = 42

day_dt = np.timedelta64(86400000000000)

price = pd.Series(
    [1.0, 2.0, 3.0, 4.0, 5.0],
    index=pd.Index(
        [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3), datetime(2020, 1, 4), datetime(2020, 1, 5)],
    ),
)
price_wide = price.vbt.tile(3, keys=["a", "b", "c"])
big_price = pd.DataFrame(np.random.uniform(size=(1000,)))
big_price.index = [datetime(2018, 1, 1) + timedelta(days=i) for i in range(1000)]
big_price_wide = big_price.vbt.tile(1000)


# ############# Global ############# #


def setup_module():
    if os.environ.get("VBT_DISABLE_CACHING", "0") == "1":
        vbt.settings.caching["disable_machinery"] = True
    vbt.settings.pbar["disable"] = True
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.portfolio["attach_call_seq"] = True


def teardown_module():
    vbt.settings.reset()


# ############# nb ############# #


def assert_same_tuple(tup1, tup2):
    for i in range(len(tup1)):
        assert tup1[i] == tup2[i] or np.isnan(tup1[i]) and np.isnan(tup2[i])


def test_execute_order_nb():
    # Errors, ignored and rejected orders
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(-100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(np.nan, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, np.inf, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, np.nan, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, np.nan, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, -10.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, 0.0, np.nan, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, size_type=-2),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, size_type=20),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, direction=-2),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, direction=20),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, -100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, direction=Direction.LongOnly),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, direction=Direction.ShortOnly),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, -10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, fees=np.inf))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, fees=np.nan))
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, fixed_fees=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, fixed_fees=np.nan),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, slippage=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, slippage=-1))
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, min_size=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, min_size=-1))
    with pytest.raises(Exception):
        nb.execute_order_nb(ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, max_size=0))
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, max_size=-10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, size_granularity=-10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, reject_prob=np.nan),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, reject_prob=-1),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, reject_prob=2),
        )

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, np.nan),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=3),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, -10.0),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=4),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, np.inf, 1100.0),
            nb.order_nb(10, 10, size_type=SizeType.Value),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, -10.0, 1100),
            nb.order_nb(10, 10, size_type=SizeType.Value),
        )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, np.nan, 1100.0),
        nb.order_nb(10, 10, size_type=SizeType.Value),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, np.inf, 1100.0),
            nb.order_nb(10, 10, size_type=SizeType.TargetValue),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(100.0, 100.0, 0.0, 100.0, -10.0, 1100),
            nb.order_nb(10, 10, size_type=SizeType.TargetValue),
        )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, np.nan, 1100.0),
        nb.order_nb(10, 10, size_type=SizeType.TargetValue),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=2),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, -10.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(np.inf, 10, direction=Direction.ShortOnly),
    )
    assert exec_state == ExecuteOrderState(cash=200.0, position=-20.0, debt=100.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, -10.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-np.inf, 10, direction=Direction.Both),
    )
    assert exec_state == ExecuteOrderState(cash=200.0, position=-20.0, debt=100.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 10.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(0, 10),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=10.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=5),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(15, 10, max_size=10, allow_partial=False),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=9),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(10, 10, reject_prob=1.0),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=10),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(0.0, 100.0, 0.0, 0.0, 10.0, 1100.0),
        nb.order_nb(10, 10, direction=Direction.LongOnly),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=100.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=7),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(0.0, 100.0, 0.0, 0.0, 10.0, 1100.0),
        nb.order_nb(10, 10, direction=Direction.Both),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=100.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=7),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(np.inf, 100, 0.0, np.inf, np.nan, 1100.0),
            nb.order_nb(np.inf, 10, direction=Direction.LongOnly),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(np.inf, 100.0, 0.0, np.inf, 10.0, 1100.0),
            nb.order_nb(np.inf, 10, direction=Direction.Both),
        )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-10, 10, direction=Direction.ShortOnly),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=0.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=8),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(np.inf, 100.0, 0.0, np.inf, 10.0, 1100.0),
            nb.order_nb(-np.inf, 10, direction=Direction.ShortOnly),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ProcessOrderState(np.inf, 100.0, 0.0, np.inf, 10.0, 1100.0),
            nb.order_nb(-np.inf, 10, direction=Direction.Both),
        )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-10, 10, direction=Direction.LongOnly),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=0.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=8),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(10, 10, fixed_fees=100),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=11),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(10, 10, min_size=100),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=12),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(100, 10, allow_partial=False),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=13),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-10, 10, min_size=100),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=12),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-200, 10, direction=Direction.LongOnly, allow_partial=False),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=13),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-10, 10, fixed_fees=1000),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=100.0, debt=0.0, free_cash=100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=11),
    )

    # Calculations
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(10, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=8.18181818181818, debt=0.0, free_cash=0.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=8.18181818181818, price=11.0, fees=10.000000000000014, side=0, status=0, status_info=-1),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(100, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=8.18181818181818, debt=0.0, free_cash=0.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=8.18181818181818, price=11.0, fees=10.000000000000014, side=0, status=0, status_info=-1),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-10, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert exec_state == ExecuteOrderState(cash=180.0, position=-10.0, debt=90.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=9.0, fees=10.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-100, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert exec_state == ExecuteOrderState(cash=909.0, position=-100.0, debt=900.0, free_cash=-891.0)
    assert_same_tuple(order_result, OrderResult(size=100.0, price=9.0, fees=91.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(10, 10, fees=-0.1, fixed_fees=-1, slippage=0.1),
    )
    assert exec_state == ExecuteOrderState(cash=2.0, position=10.0, debt=0.0, free_cash=2.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=11.0, fees=-12.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(10, 0, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert exec_state == ExecuteOrderState(cash=99.0, position=10.0, debt=0.0, free_cash=99.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=0.0, fees=1.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 10.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-10, 0, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert exec_state == ExecuteOrderState(cash=99.0, position=0.0, debt=0.0, free_cash=99.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=0.0, fees=1.0, side=1, status=0, status_info=-1))

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(10, 10, size_type=SizeType.TargetAmount),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=10.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-10, 10, size_type=SizeType.TargetAmount),
    )
    assert exec_state == ExecuteOrderState(cash=200.0, position=-10.0, debt=100.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(100, 10, size_type=SizeType.Value),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=10.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-100, 10, size_type=SizeType.Value),
    )
    assert exec_state == ExecuteOrderState(cash=200.0, position=-10.0, debt=100.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(100, 10, size_type=SizeType.TargetValue),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=10.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-100, 10, size_type=SizeType.TargetValue),
    )
    assert exec_state == ExecuteOrderState(cash=200.0, position=-10.0, debt=100.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=10.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-1, 10, size_type=SizeType.TargetPercent),
    )
    assert exec_state == ExecuteOrderState(cash=200.0, position=-10.0, debt=100.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, 5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=10.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, 5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=25.0, position=7.5, debt=0.0, free_cash=25.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, 5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=125.0, position=-2.5, debt=25.0, free_cash=75.0)
    assert_same_tuple(order_result, OrderResult(size=7.5, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, 5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=200.0, position=-10.0, debt=100.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=15.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=5.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=25.0, position=2.5, debt=0.0, free_cash=25.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=75.0, position=-2.5, debt=25.0, free_cash=25.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=-5.0, debt=50.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, -5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=0.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, -5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=25.0, position=-2.5, debt=0.0, free_cash=25.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, -5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=75.0, position=-7.5, debt=25.0, free_cash=25.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(50.0, -5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert exec_state == ExecuteOrderState(cash=100.0, position=-10.0, debt=50.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(np.inf, 10),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=10.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, -5.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(np.inf, 10),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=5.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-np.inf, 10),
    )
    assert exec_state == ExecuteOrderState(cash=200.0, position=-10.0, debt=100.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(150.0, -5.0, 0.0, 150.0, 10.0, 100.0),
        nb.order_nb(-np.inf, 10),
    )
    assert exec_state == ExecuteOrderState(cash=300.0, position=-20.0, debt=150.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=15.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(10, 10, lock_cash=True),
    )
    assert exec_state == ExecuteOrderState(cash=50.0, position=5.0, debt=0.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(1000.0, -5.0, 50.0, 50.0, 10.0, 100.0),
        nb.order_nb(10, 17.5, lock_cash=True),
    )
    assert exec_state == ExecuteOrderState(cash=850.0, position=3.571428571428571, debt=0.0, free_cash=0.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=8.571428571428571, price=17.5, fees=0.0, side=0, status=0, status_info=-1),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(100.0, -5.0, 50.0, 50.0, 10.0, 100.0),
        nb.order_nb(10, 100, lock_cash=True),
    )
    assert exec_state == ExecuteOrderState(cash=37.5, position=-4.375, debt=43.75, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=0.625, price=100.0, fees=0.0, side=0, status=0, status_info=-1))

    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(0.0, 10.0, 0.0, -50.0, 10.0, 100.0),
        nb.order_nb(-20, 10, lock_cash=True),
    )
    assert exec_state == ExecuteOrderState(cash=150.0, position=-5.0, debt=50.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=15.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(0.0, 1.0, 0.0, -50.0, 10.0, 100.0),
        nb.order_nb(-10, 10, lock_cash=True),
    )
    assert exec_state == ExecuteOrderState(cash=10.0, position=0.0, debt=0.0, free_cash=-40.0)
    assert_same_tuple(order_result, OrderResult(size=1.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(0.0, 0.0, 0.0, -100.0, 10.0, 100.0),
        nb.order_nb(-10, 10, lock_cash=True),
    )
    assert exec_state == ExecuteOrderState(cash=0.0, position=0.0, debt=0.0, free_cash=-100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=6),
    )
    exec_state, order_result = nb.execute_order_nb(
        ProcessOrderState(0.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-20, 10, fees=0.1, slippage=0.1, fixed_fees=1.0, lock_cash=True),
    )
    assert exec_state == ExecuteOrderState(cash=80.0, position=-10.0, debt=90.0, free_cash=0.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=9.0, fees=10.0, side=1, status=0, status_info=-1))


def test_build_call_seq_nb():
    group_lens = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(
        build_call_seq_nb((10, 10), group_lens, CallSeqType.Default),
        build_call_seq((10, 10), group_lens, CallSeqType.Default),
    )
    np.testing.assert_array_equal(
        build_call_seq_nb((10, 10), group_lens, CallSeqType.Reversed),
        build_call_seq((10, 10), group_lens, CallSeqType.Reversed),
    )
    set_seed(seed)
    out1 = build_call_seq_nb((10, 10), group_lens, CallSeqType.Random)
    set_seed(seed)
    out2 = build_call_seq((10, 10), group_lens, CallSeqType.Random)
    np.testing.assert_array_equal(out1, out2)


# ############# from_orders ############# #

order_size = pd.Series([np.inf, -np.inf, np.nan, np.inf, -np.inf], index=price.index)
order_size_wide = order_size.vbt.tile(3, keys=["a", "b", "c"])
order_size_one = pd.Series([1, -1, np.nan, 1, -1], index=price.index)


def from_orders_both(close=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(close, size, direction="both", **kwargs)


def from_orders_longonly(close=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(close, size, direction="longonly", **kwargs)


def from_orders_shortonly(close=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(close, size, direction="shortonly", **kwargs)


class TestFromOrders:
    def test_one_column(self):
        assert_records_close(
            from_orders_both().order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 1, 200.0, 2.0, 0.0, 1), (2, 0, 3, 100.0, 4.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly().order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly().order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 0, 1, 100.0, 2.0, 0.0, 0)], dtype=order_dt),
        )
        pf = from_orders_both()
        pd.testing.assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        pd.testing.assert_index_equal(pf.wrapper.columns, pd.Int64Index([0], dtype="int64"))
        assert pf.wrapper.ndim == 1
        assert pf.wrapper.freq == day_dt
        assert pf.wrapper.grouper.group_by is None

    def test_multiple_columns(self):
        assert_records_close(
            from_orders_both(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 2.0, 0.0, 1),
                    (2, 1, 3, 50.0, 4.0, 0.0, 0),
                    (3, 1, 4, 50.0, 5.0, 0.0, 1),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 100.0, 2.0, 0.0, 1),
                    (2, 2, 3, 50.0, 4.0, 0.0, 0),
                    (3, 2, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(close=price_wide).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (1, 0, 1, 100.0, 2.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (1, 1, 1, 100.0, 2.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 1),
                    (1, 2, 1, 100.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pf = from_orders_both(close=price_wide)
        pd.testing.assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        pd.testing.assert_index_equal(pf.wrapper.columns, pd.Index(["a", "b", "c"], dtype="object"))
        assert pf.wrapper.ndim == 2
        assert pf.wrapper.freq == day_dt
        assert pf.wrapper.grouper.group_by is None

    def test_size_inf(self):
        assert_records_close(
            from_orders_both(size=[[np.inf, -np.inf]]).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0), (0, 1, 0, 100.0, 1.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_longonly(size=[[np.inf, -np.inf]]).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_shortonly(size=[[np.inf, -np.inf]]).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 1)], dtype=order_dt),
        )

    def test_size_granularity(self):
        assert_records_close(
            from_orders_both(fees=0.1, fixed_fees=0.1, size_granularity=1).order_records,
            np.array(
                [(0, 0, 0, 90.0, 1.0, 9.1, 0), (1, 0, 1, 164.0, 2.0, 32.9, 1), (2, 0, 3, 67.0, 4.0, 26.9, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(fees=0.1, fixed_fees=0.1, size_granularity=1).order_records,
            np.array(
                [
                    (0, 0, 0, 90.0, 1.0, 9.1, 0),
                    (1, 0, 1, 90.0, 2.0, 18.1, 1),
                    (2, 0, 3, 36.0, 4.0, 14.5, 0),
                    (3, 0, 4, 36.0, 5.0, 18.1, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(fees=0.1, fixed_fees=0.1, size_granularity=1).order_records,
            np.array([(0, 0, 0, 90.0, 1.0, 9.1, 1), (1, 0, 1, 82.0, 2.0, 16.5, 0)], dtype=order_dt),
        )

    def test_price(self):
        assert_records_close(
            from_orders_both(price=price * 1.01).order_records,
            np.array(
                [
                    (0, 0, 0, 99.00990099009901, 1.01, 0.0, 0),
                    (1, 0, 1, 198.01980198019803, 2.02, 0.0, 1),
                    (2, 0, 3, 99.00990099009901, 4.04, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(price=price * 1.01).order_records,
            np.array(
                [
                    (0, 0, 0, 99.00990099009901, 1.01, 0.0, 0),
                    (1, 0, 1, 99.00990099009901, 2.02, 0.0, 1),
                    (2, 0, 3, 49.504950495049506, 4.04, 0.0, 0),
                    (3, 0, 4, 49.504950495049506, 5.05, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(price=price * 1.01).order_records,
            np.array(
                [(0, 0, 0, 99.00990099009901, 1.01, 0.0, 1), (1, 0, 1, 99.00990099009901, 2.02, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(price=np.inf).order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 1, 200.0, 2.0, 0.0, 1), (2, 0, 3, 100.0, 4.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(price=np.inf).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(price=np.inf).order_records,
            np.array([(0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 0, 1, 100.0, 2.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_both(price=-np.inf, open=price.shift(1)).order_records,
            np.array([(0, 0, 1, 100.0, 1.0, 0.0, 1), (1, 0, 3, 66.66666666666667, 3.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_longonly(price=-np.inf, open=price.shift(1)).order_records,
            np.array(
                [(0, 0, 3, 33.333333333333336, 3.0, 0.0, 0), (1, 0, 4, 33.333333333333336, 4.0, 0.0, 1)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(price=-np.inf, open=price.shift(1)).order_records,
            np.array(
                [(0, 0, 3, 33.333333333333336, 3.0, 0.0, 1), (1, 0, 4, 33.333333333333336, 4.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )

    def test_price_area(self):
        assert_records_close(
            from_orders_both(
                open=2,
                high=4,
                low=1,
                close=3,
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
            from_orders_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
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
            from_orders_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
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
            from_orders_both(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
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
            from_orders_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
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
            from_orders_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="cap",
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
            from_orders_longonly(
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
            from_orders_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=np.inf,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_orders_longonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=5,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_orders_shortonly(
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
            from_orders_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=np.inf,
                size=1,
                slippage=0.1,
            )
        with pytest.raises(Exception):
            from_orders_shortonly(
                open=2,
                high=4,
                low=1,
                close=3,
                price_area_vio_mode="error",
                price=5,
                size=1,
                slippage=0.1,
            )

    def test_val_price(self):
        price_nan = pd.Series([1, 2, np.nan, 4, 5], index=price.index)
        assert_records_close(
            from_orders_both(close=price_nan, size=order_size_one, val_price=np.inf, size_type="value").order_records,
            from_orders_both(close=price_nan, size=order_size_one, val_price=price, size_type="value").order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=price,
                size_type="value",
            ).order_records,
        )
        shift_price = price_nan.ffill().shift(1)
        assert_records_close(
            from_orders_both(close=price_nan, size=order_size_one, val_price=-np.inf, size_type="value").order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        price_all_nan = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], index=price.index)
        assert_records_close(
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=-np.inf,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_all_nan,
                size_type="value",
                ffill_val_price=False,
            ).order_records,
        )
        assert_records_close(
            from_orders_both(close=price_nan, size=order_size_one, val_price=np.nan, size_type="value").order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=shift_price,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                close=price_nan,
                open=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_both(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_longonly(
                close=price_nan,
                open=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_longonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
            ).order_records,
        )
        assert_records_close(
            from_orders_shortonly(
                close=price_nan,
                open=price_nan,
                size=order_size_one,
                val_price=np.nan,
                size_type="value",
            ).order_records,
            from_orders_shortonly(
                close=price_nan,
                size=order_size_one,
                val_price=price_nan,
                size_type="value",
            ).order_records,
        )

    def test_fees(self):
        assert_records_close(
            from_orders_both(size=order_size_one, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 1, 1.0, 2.0, -0.2, 1),
                    (2, 0, 3, 1.0, 4.0, -0.4, 0),
                    (3, 0, 4, 1.0, 5.0, -0.5, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 1, 1.0, 2.0, 0.2, 1),
                    (2, 2, 3, 1.0, 4.0, 0.4, 0),
                    (3, 2, 4, 1.0, 5.0, 0.5, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 1, 1.0, 2.0, 2.0, 1),
                    (2, 3, 3, 1.0, 4.0, 4.0, 0),
                    (3, 3, 4, 1.0, 5.0, 5.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 1, 1.0, 2.0, -0.2, 1),
                    (2, 0, 3, 1.0, 4.0, -0.4, 0),
                    (3, 0, 4, 1.0, 5.0, -0.5, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 1, 1.0, 2.0, 0.2, 1),
                    (2, 2, 3, 1.0, 4.0, 0.4, 0),
                    (3, 2, 4, 1.0, 5.0, 0.5, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 1, 1.0, 2.0, 2.0, 1),
                    (2, 3, 3, 1.0, 4.0, 4.0, 0),
                    (3, 3, 4, 1.0, 5.0, 5.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 1),
                    (1, 0, 1, 1.0, 2.0, -0.2, 0),
                    (2, 0, 3, 1.0, 4.0, -0.4, 1),
                    (3, 0, 4, 1.0, 5.0, -0.5, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 1.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.1, 1),
                    (1, 2, 1, 1.0, 2.0, 0.2, 0),
                    (2, 2, 3, 1.0, 4.0, 0.4, 1),
                    (3, 2, 4, 1.0, 5.0, 0.5, 0),
                    (0, 3, 0, 1.0, 1.0, 1.0, 1),
                    (1, 3, 1, 1.0, 2.0, 2.0, 0),
                    (2, 3, 3, 1.0, 4.0, 4.0, 1),
                    (3, 3, 4, 1.0, 5.0, 5.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_fixed_fees(self):
        assert_records_close(
            from_orders_both(size=order_size_one, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 1, 1.0, 2.0, -0.1, 1),
                    (2, 0, 3, 1.0, 4.0, -0.1, 0),
                    (3, 0, 4, 1.0, 5.0, -0.1, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 1, 1.0, 2.0, 0.1, 1),
                    (2, 2, 3, 1.0, 4.0, 0.1, 0),
                    (3, 2, 4, 1.0, 5.0, 0.1, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 1, 1.0, 2.0, 1.0, 1),
                    (2, 3, 3, 1.0, 4.0, 1.0, 0),
                    (3, 3, 4, 1.0, 5.0, 1.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 0),
                    (1, 0, 1, 1.0, 2.0, -0.1, 1),
                    (2, 0, 3, 1.0, 4.0, -0.1, 0),
                    (3, 0, 4, 1.0, 5.0, -0.1, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.1, 0),
                    (1, 2, 1, 1.0, 2.0, 0.1, 1),
                    (2, 2, 3, 1.0, 4.0, 0.1, 0),
                    (3, 2, 4, 1.0, 5.0, 0.1, 1),
                    (0, 3, 0, 1.0, 1.0, 1.0, 0),
                    (1, 3, 1, 1.0, 2.0, 1.0, 1),
                    (2, 3, 3, 1.0, 4.0, 1.0, 0),
                    (3, 3, 4, 1.0, 5.0, 1.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, fixed_fees=[[-0.1, 0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, -0.1, 1),
                    (1, 0, 1, 1.0, 2.0, -0.1, 0),
                    (2, 0, 3, 1.0, 4.0, -0.1, 1),
                    (3, 0, 4, 1.0, 5.0, -0.1, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 1.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.1, 1),
                    (1, 2, 1, 1.0, 2.0, 0.1, 0),
                    (2, 2, 3, 1.0, 4.0, 0.1, 1),
                    (3, 2, 4, 1.0, 5.0, 0.1, 0),
                    (0, 3, 0, 1.0, 1.0, 1.0, 1),
                    (1, 3, 1, 1.0, 2.0, 1.0, 0),
                    (2, 3, 3, 1.0, 4.0, 1.0, 1),
                    (3, 3, 4, 1.0, 5.0, 1.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_slippage(self):
        assert_records_close(
            from_orders_both(size=order_size_one, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.1, 0.0, 0),
                    (1, 1, 1, 1.0, 1.8, 0.0, 1),
                    (2, 1, 3, 1.0, 4.4, 0.0, 0),
                    (3, 1, 4, 1.0, 4.5, 0.0, 1),
                    (0, 2, 0, 1.0, 2.0, 0.0, 0),
                    (1, 2, 1, 1.0, 0.0, 0.0, 1),
                    (2, 2, 3, 1.0, 8.0, 0.0, 0),
                    (3, 2, 4, 1.0, 0.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.1, 0.0, 0),
                    (1, 1, 1, 1.0, 1.8, 0.0, 1),
                    (2, 1, 3, 1.0, 4.4, 0.0, 0),
                    (3, 1, 4, 1.0, 4.5, 0.0, 1),
                    (0, 2, 0, 1.0, 2.0, 0.0, 0),
                    (1, 2, 1, 1.0, 0.0, 0.0, 1),
                    (2, 2, 3, 1.0, 8.0, 0.0, 0),
                    (3, 2, 4, 1.0, 0.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, slippage=[[0.0, 0.1, 1.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 1, 1.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1.0, 4.0, 0.0, 1),
                    (3, 0, 4, 1.0, 5.0, 0.0, 0),
                    (0, 1, 0, 1.0, 0.9, 0.0, 1),
                    (1, 1, 1, 1.0, 2.2, 0.0, 0),
                    (2, 1, 3, 1.0, 3.6, 0.0, 1),
                    (3, 1, 4, 1.0, 5.5, 0.0, 0),
                    (0, 2, 0, 1.0, 0.0, 0.0, 1),
                    (1, 2, 1, 1.0, 4.0, 0.0, 0),
                    (2, 2, 3, 1.0, 0.0, 0.0, 1),
                    (3, 2, 4, 1.0, 10.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_min_size(self):
        assert_records_close(
            from_orders_both(size=order_size_one, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, min_size=[[0.0, 1.0, 2.0]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 1, 1.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1.0, 4.0, 0.0, 1),
                    (3, 0, 4, 1.0, 5.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 1.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_max_size(self):
        assert_records_close(
            from_orders_both(size=order_size_one, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 0),
                    (1, 0, 1, 0.5, 2.0, 0.0, 1),
                    (2, 0, 3, 0.5, 4.0, 0.0, 0),
                    (3, 0, 4, 0.5, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 1, 1.0, 2.0, 0.0, 1),
                    (2, 2, 3, 1.0, 4.0, 0.0, 0),
                    (3, 2, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 0),
                    (1, 0, 1, 0.5, 2.0, 0.0, 1),
                    (2, 0, 3, 0.5, 4.0, 0.0, 0),
                    (3, 0, 4, 0.5, 5.0, 0.0, 1),
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 3, 1.0, 4.0, 0.0, 0),
                    (3, 1, 4, 1.0, 5.0, 0.0, 1),
                    (0, 2, 0, 1.0, 1.0, 0.0, 0),
                    (1, 2, 1, 1.0, 2.0, 0.0, 1),
                    (2, 2, 3, 1.0, 4.0, 0.0, 0),
                    (3, 2, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, max_size=[[0.5, 1.0, np.inf]]).order_records,
            np.array(
                [
                    (0, 0, 0, 0.5, 1.0, 0.0, 1),
                    (1, 0, 1, 0.5, 2.0, 0.0, 0),
                    (2, 0, 3, 0.5, 4.0, 0.0, 1),
                    (3, 0, 4, 0.5, 5.0, 0.0, 0),
                    (0, 1, 0, 1.0, 1.0, 0.0, 1),
                    (1, 1, 1, 1.0, 2.0, 0.0, 0),
                    (2, 1, 3, 1.0, 4.0, 0.0, 1),
                    (3, 1, 4, 1.0, 5.0, 0.0, 0),
                    (0, 2, 0, 1.0, 1.0, 0.0, 1),
                    (1, 2, 1, 1.0, 2.0, 0.0, 0),
                    (2, 2, 3, 1.0, 4.0, 0.0, 1),
                    (3, 2, 4, 1.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_reject_prob(self):
        assert_records_close(
            from_orders_both(size=order_size_one, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 1, 1.0, 2.0, 0.0, 1),
                    (1, 1, 3, 1.0, 4.0, 0.0, 0),
                    (2, 1, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1.0, 2.0, 0.0, 1),
                    (2, 0, 3, 1.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1.0, 5.0, 0.0, 1),
                    (0, 1, 3, 1.0, 4.0, 0.0, 0),
                    (1, 1, 4, 1.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, reject_prob=[[0.0, 0.5, 1.0]], seed=42).order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 1, 1.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1.0, 4.0, 0.0, 1),
                    (3, 0, 4, 1.0, 5.0, 0.0, 0),
                    (0, 1, 3, 1.0, 4.0, 0.0, 1),
                    (1, 1, 4, 1.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_lock_cash(self):
        pf = vbt.Portfolio.from_orders(
            pd.Series([1, 1]),
            pd.DataFrame([[-25, -25], [np.inf, np.inf]]),
            group_by=True,
            cash_sharing=True,
            lock_cash=False,
            fees=0.01,
            fixed_fees=1.0,
            slippage=0.01,
        )
        np.testing.assert_array_equal(pf.asset_flow.values, np.array([[-25.0, -25.0], [143.12812469365747, 0.0]]))
        pf = vbt.Portfolio.from_orders(
            pd.Series([1, 1]),
            pd.DataFrame([[-25, -25], [np.inf, np.inf]]),
            group_by=True,
            cash_sharing=True,
            lock_cash=True,
            fees=0.01,
            fixed_fees=1.0,
            slippage=0.01,
        )
        np.testing.assert_array_equal(
            pf.asset_flow.values,
            np.array([[-25.0, -25.0], [94.6034702480149, 47.54435839623566]]),
        )
        pf = vbt.Portfolio.from_orders(
            pd.Series([1, 100]),
            pd.DataFrame([[-25, -25], [np.inf, np.inf]]),
            group_by=True,
            cash_sharing=True,
            lock_cash=False,
            fees=0.01,
            fixed_fees=1.0,
            slippage=0.01,
        )
        np.testing.assert_array_equal(pf.asset_flow.values, np.array([[-25.0, -25.0], [1.4312812469365748, 0.0]]))
        pf = vbt.Portfolio.from_orders(
            pd.Series([1, 100]),
            pd.DataFrame([[-25, -25], [np.inf, np.inf]]),
            group_by=True,
            cash_sharing=True,
            lock_cash=True,
            fees=0.01,
            fixed_fees=1.0,
            slippage=0.01,
        )
        np.testing.assert_array_equal(pf.asset_flow.values, np.array([[-25.0, -25.0], [0.4699090272918124, 0.0]]))
        pf = from_orders_both(size=order_size_one * 1000, lock_cash=[[False, True]])
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1000.0, 2.0, 0.0, 1),
                    (2, 0, 3, 500.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1000.0, 5.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.get_cash(free=True).values,
            np.array([[0.0, 0.0], [-1600.0, 0.0], [-1600.0, 0.0], [-1600.0, 0.0], [-6600.0, 0.0]]),
        )
        pf = from_orders_longonly(size=order_size_one * 1000, lock_cash=[[False, True]])
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 2.0, 0.0, 1),
                    (2, 1, 3, 50.0, 4.0, 0.0, 0),
                    (3, 1, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.get_cash(free=True).values,
            np.array([[0.0, 0.0], [200.0, 200.0], [200.0, 200.0], [0.0, 0.0], [250.0, 250.0]]),
        )
        pf = from_orders_shortonly(size=order_size_one * 1000, lock_cash=[[False, True]])
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 1000.0, 1.0, 0.0, 1),
                    (1, 0, 1, 550.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1000.0, 4.0, 0.0, 1),
                    (3, 0, 4, 800.0, 5.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (1, 1, 1, 100.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.get_cash(free=True).values,
            np.array([[-900.0, 0.0], [-900.0, 0.0], [-900.0, 0.0], [-4900.0, 0.0], [-3989.6551724137926, 0.0]]),
        )

    def test_allow_partial(self):
        assert_records_close(
            from_orders_both(size=order_size_one * 1000, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1000.0, 2.0, 0.0, 1),
                    (2, 0, 3, 500.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1000.0, 5.0, 0.0, 1),
                    (0, 1, 1, 1000.0, 2.0, 0.0, 1),
                    (1, 1, 4, 1000.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one * 1000, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one * 1000, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 1000.0, 1.0, 0.0, 1),
                    (1, 0, 1, 550.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1000.0, 4.0, 0.0, 1),
                    (3, 0, 4, 800.0, 5.0, 0.0, 0),
                    (0, 1, 0, 1000.0, 1.0, 0.0, 1),
                    (1, 1, 3, 1000.0, 4.0, 0.0, 1),
                    (2, 1, 4, 1000.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 100.0, 2.0, 0.0, 1),
                    (2, 1, 3, 50.0, 4.0, 0.0, 0),
                    (3, 1, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 1),
                    (1, 0, 1, 100.0, 2.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 1),
                    (1, 1, 1, 100.0, 2.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_raise_reject(self):
        assert_records_close(
            from_orders_both(size=order_size_one * 1000, allow_partial=True, raise_reject=True).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 1000.0, 2.0, 0.0, 1),
                    (2, 0, 3, 500.0, 4.0, 0.0, 0),
                    (3, 0, 4, 1000.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one * 1000, allow_partial=True, raise_reject=True).order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 100.0, 2.0, 0.0, 1),
                    (2, 0, 3, 50.0, 4.0, 0.0, 0),
                    (3, 0, 4, 50.0, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one * 1000, allow_partial=True, raise_reject=True).order_records,
            np.array(
                [
                    (0, 0, 0, 1000.0, 1.0, 0.0, 1),
                    (1, 0, 1, 550.0, 2.0, 0.0, 0),
                    (2, 0, 3, 1000.0, 4.0, 0.0, 1),
                    (3, 0, 4, 800.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        with pytest.raises(Exception):
            from_orders_both(size=order_size_one * 1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception):
            from_orders_longonly(size=order_size_one * 1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception):
            from_orders_shortonly(size=order_size_one * 1000, allow_partial=False, raise_reject=True).order_records

    def test_log(self):
        assert_records_close(
            from_orders_both(log=True).log_records,
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
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        0.0,
                        100.0,
                        0.0,
                        0.0,
                        2.0,
                        200.0,
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
                        400.0,
                        -100.0,
                        200.0,
                        0.0,
                        2.0,
                        200.0,
                        200.0,
                        2.0,
                        0.0,
                        1,
                        0,
                        -1,
                        1,
                    ),
                    (
                        2,
                        0,
                        0,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        400.0,
                        -100.0,
                        200.0,
                        0.0,
                        3.0,
                        100.0,
                        np.nan,
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
                        400.0,
                        -100.0,
                        200.0,
                        0.0,
                        3.0,
                        100.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        -1,
                        1,
                        0,
                        -1,
                    ),
                    (
                        3,
                        0,
                        0,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        400.0,
                        -100.0,
                        200.0,
                        0.0,
                        4.0,
                        0.0,
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
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        100.0,
                        4.0,
                        0.0,
                        0,
                        0,
                        -1,
                        2,
                    ),
                    (
                        4,
                        0,
                        0,
                        4,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        5.0,
                        0.0,
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
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        5.0,
                        0.0,
                        np.nan,
                        np.nan,
                        np.nan,
                        -1,
                        2,
                        6,
                        -1,
                    ),
                ],
                dtype=log_dt,
            ),
        )

    def test_group_by(self):
        pf = from_orders_both(close=price_wide, group_by=np.array([0, 0, 1]))
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pd.testing.assert_index_equal(pf.wrapper.grouper.group_by, pd.Int64Index([0, 0, 1], dtype="int64"))
        pd.testing.assert_series_equal(
            pf.init_cash,
            pd.Series([200.0, 100.0], index=pd.Int64Index([0, 1], dtype="int64")).rename("init_cash"),
        )
        assert not pf.cash_sharing

    def test_cash_sharing(self):
        pf = from_orders_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pd.testing.assert_index_equal(pf.wrapper.grouper.group_by, pd.Int64Index([0, 0, 1], dtype="int64"))
        pd.testing.assert_series_equal(
            pf.init_cash,
            pd.Series([100.0, 100.0], index=pd.Int64Index([0, 1], dtype="int64")).rename("init_cash"),
        )
        assert pf.cash_sharing
        with pytest.raises(Exception):
            pf.regroup(group_by=False)

    def test_call_seq(self):
        pf = from_orders_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        )
        pf = from_orders_both(close=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True, call_seq="reversed")
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        pf = from_orders_both(
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
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 3, 100.0, 4.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 3, 100.0, 4.0, 0.0, 0),
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
            size=pd.DataFrame(
                [
                    [0.0, 0.0, np.inf],
                    [0.0, np.inf, -np.inf],
                    [np.inf, -np.inf, 0.0],
                    [-np.inf, 0.0, np.inf],
                    [0.0, np.inf, -np.inf],
                ]
            ),
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq="auto",
        )
        pf = from_orders_both(**kwargs)
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
        pf = from_orders_longonly(**kwargs)
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
        pf = from_orders_shortonly(**kwargs)
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
            np.array([[2, 0, 1], [1, 0, 2], [0, 2, 1], [2, 1, 0], [1, 0, 2]]),
        )

    def test_value(self):
        assert_records_close(
            from_orders_both(size=order_size_one, size_type="value").order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 0.5, 2.0, 0.0, 1),
                    (2, 0, 3, 0.25, 4.0, 0.0, 0),
                    (3, 0, 4, 0.2, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=order_size_one, size_type="value").order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 0.5, 2.0, 0.0, 1),
                    (2, 0, 3, 0.25, 4.0, 0.0, 0),
                    (3, 0, 4, 0.2, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=order_size_one, size_type="value").order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 1),
                    (1, 0, 1, 0.5, 2.0, 0.0, 0),
                    (2, 0, 3, 0.25, 4.0, 0.0, 1),
                    (3, 0, 4, 0.2, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_target_amount(self):
        assert_records_close(
            from_orders_both(size=[[75.0, -75.0]], size_type="targetamount").order_records,
            np.array([(0, 0, 0, 75.0, 1.0, 0.0, 0), (0, 1, 0, 75.0, 1.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_longonly(size=[[75.0, -75.0]], size_type="targetamount").order_records,
            np.array([(0, 0, 0, 75.0, 1.0, 0.0, 0)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_shortonly(size=[[75.0, -75.0]], size_type="targetamount").order_records,
            np.array([(0, 0, 0, 75.0, 1.0, 0.0, 1)], dtype=order_dt),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=75.0,
                size_type="targetamount",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array([(0, 0, 0, 75.0, 1.0, 0.0, 0), (0, 1, 0, 25.0, 1.0, 0.0, 0)], dtype=order_dt),
        )

    def test_target_value(self):
        assert_records_close(
            from_orders_both(size=[[50.0, -50.0]], size_type="targetvalue").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 25.0, 2.0, 0.0, 1),
                    (2, 0, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (3, 0, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (4, 0, 4, 2.5, 5.0, 0.0, 1),
                    (0, 1, 0, 50.0, 1.0, 0.0, 1),
                    (1, 1, 1, 25.0, 2.0, 0.0, 0),
                    (2, 1, 2, 8.333333333333332, 3.0, 0.0, 0),
                    (3, 1, 3, 4.166666666666668, 4.0, 0.0, 0),
                    (4, 1, 4, 2.5, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=[[50.0, -50.0]], size_type="targetvalue").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 25.0, 2.0, 0.0, 1),
                    (2, 0, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (3, 0, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (4, 0, 4, 2.5, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=[[50.0, -50.0]], size_type="targetvalue").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 1),
                    (1, 0, 1, 25.0, 2.0, 0.0, 0),
                    (2, 0, 2, 8.333333333333332, 3.0, 0.0, 0),
                    (3, 0, 3, 4.166666666666668, 4.0, 0.0, 0),
                    (4, 0, 4, 2.5, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=50.0,
                size_type="targetvalue",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 25.0, 2.0, 0.0, 1),
                    (2, 0, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (3, 0, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (4, 0, 4, 2.5, 5.0, 0.0, 1),
                    (0, 1, 0, 50.0, 1.0, 0.0, 0),
                    (1, 1, 1, 25.0, 2.0, 0.0, 1),
                    (2, 1, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (3, 1, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (4, 1, 4, 2.5, 5.0, 0.0, 1),
                    (0, 2, 1, 25.0, 2.0, 0.0, 0),
                    (1, 2, 2, 8.333333333333332, 3.0, 0.0, 1),
                    (2, 2, 3, 4.166666666666668, 4.0, 0.0, 1),
                    (3, 2, 4, 2.5, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    def test_target_percent(self):
        assert_records_close(
            from_orders_both(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 1),
                    (2, 0, 2, 6.25, 3.0, 0.0, 1),
                    (3, 0, 3, 3.90625, 4.0, 0.0, 1),
                    (4, 0, 4, 2.734375, 5.0, 0.0, 1),
                    (0, 1, 0, 50.0, 1.0, 0.0, 1),
                    (1, 1, 1, 37.5, 2.0, 0.0, 0),
                    (2, 1, 2, 6.25, 3.0, 0.0, 0),
                    (3, 1, 3, 2.34375, 4.0, 0.0, 0),
                    (4, 1, 4, 1.171875, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 1),
                    (2, 0, 2, 6.25, 3.0, 0.0, 1),
                    (3, 0, 3, 3.90625, 4.0, 0.0, 1),
                    (4, 0, 4, 2.734375, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=[[0.5, -0.5]], size_type="targetpercent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 1),
                    (1, 0, 1, 37.5, 2.0, 0.0, 0),
                    (2, 0, 2, 6.25, 3.0, 0.0, 0),
                    (3, 0, 3, 2.34375, 4.0, 0.0, 0),
                    (4, 0, 4, 1.171875, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=0.5,
                size_type="targetpercent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array([(0, 0, 0, 50.0, 1.0, 0.0, 0), (0, 1, 0, 50.0, 1.0, 0.0, 0)], dtype=order_dt),
        )

    def test_update_value(self):
        assert_records_close(
            from_orders_both(
                size=0.5,
                size_type="targetpercent",
                fees=0.01,
                slippage=0.01,
                update_value=False,
            ).order_records,
            from_orders_both(
                size=0.5,
                size_type="targetpercent",
                fees=0.01,
                slippage=0.01,
                update_value=True,
            ).order_records,
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=0.5,
                size_type="targetpercent",
                fees=0.01,
                slippage=0.01,
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                update_value=False,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.01, 0.505, 0),
                    (1, 0, 1, 0.9851975296539592, 1.98, 0.019506911087148394, 1),
                    (2, 0, 2, 0.019315704924103727, 2.9699999999999998, 0.0005736764362458806, 1),
                    (3, 0, 3, 0.00037870218456959037, 3.96, 1.4996606508955778e-05, 1),
                    (4, 0, 4, 7.424805112066224e-06, 4.95, 3.675278530472781e-07, 1),
                    (0, 1, 0, 48.02960494069208, 1.01, 0.485099009900992, 0),
                    (1, 1, 1, 0.9465661198057499, 2.02, 0.019120635620076154, 0),
                    (2, 1, 2, 0.018558300554959377, 3.0300000000000002, 0.0005623165068152705, 0),
                    (3, 1, 3, 0.0003638525743521767, 4.04, 1.4699644003827875e-05, 0),
                    (4, 1, 4, 7.133664827307231e-06, 5.05, 3.6025007377901643e-07, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=0.5,
                size_type="targetpercent",
                fees=0.01,
                slippage=0.01,
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                update_value=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.01, 0.505, 0),
                    (1, 0, 1, 0.9851975296539592, 1.98, 0.019506911087148394, 1),
                    (2, 0, 2, 0.019315704924103727, 2.9699999999999998, 0.0005736764362458806, 1),
                    (3, 0, 3, 0.0005670876809631409, 3.96, 2.2456672166140378e-05, 1),
                    (4, 0, 4, 1.8523501267964093e-05, 4.95, 9.169133127642227e-07, 1),
                    (0, 1, 0, 48.02960494069208, 1.01, 0.485099009900992, 0),
                    (1, 1, 1, 0.7303208018821721, 2.02, 0.014752480198019875, 0),
                    (2, 1, 2, 0.009608602243410758, 2.9699999999999998, 0.00028537548662929945, 1),
                    (3, 1, 3, 0.00037770350099464167, 3.96, 1.4957058639387809e-05, 1),
                    (4, 1, 4, 1.2972670177191503e-05, 4.95, 6.421471737709794e-07, 1),
                    (0, 2, 1, 0.21624531792357785, 2.02, 0.0043681554220562635, 0),
                    (1, 2, 2, 0.02779013180558861, 3.0300000000000002, 0.0008420409937093393, 0),
                    (2, 2, 3, 0.0009077441794302741, 4.04, 3.6672864848982974e-05, 0),
                    (3, 2, 4, 3.0261148547590434e-05, 5.05, 1.5281880016533242e-06, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_percent(self):
        assert_records_close(
            from_orders_both(size=[[0.5, -0.5]], size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 0),
                    (2, 0, 2, 4.16666667, 3.0, 0.0, 0),
                    (3, 0, 3, 1.5625, 4.0, 0.0, 0),
                    (4, 0, 4, 0.625, 5.0, 0.0, 0),
                    (0, 1, 0, 50.0, 1.0, 0.0, 1),
                    (1, 1, 1, 12.5, 2.0, 0.0, 1),
                    (2, 1, 2, 4.16666667, 3.0, 0.0, 1),
                    (3, 1, 3, 1.5625, 4.0, 0.0, 1),
                    (4, 1, 4, 0.625, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_longonly(size=[[0.5, -0.5]], size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 12.5, 2.0, 0.0, 0),
                    (2, 0, 2, 4.16666667, 3.0, 0.0, 0),
                    (3, 0, 3, 1.5625, 4.0, 0.0, 0),
                    (4, 0, 4, 0.625, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_shortonly(size=[[0.5, -0.5]], size_type="percent").order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 1),
                    (1, 0, 1, 12.5, 2.0, 0.0, 1),
                    (2, 0, 2, 4.16666667, 3.0, 0.0, 1),
                    (3, 0, 3, 1.5625, 4.0, 0.0, 1),
                    (4, 0, 4, 0.625, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            from_orders_both(
                close=price_wide,
                size=0.5,
                size_type="percent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
            ).order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 1.0, 0.0, 0),
                    (1, 0, 1, 3.125, 2.0, 0.0, 0),
                    (2, 0, 2, 0.2604166666666667, 3.0, 0.0, 0),
                    (3, 0, 3, 0.0244140625, 4.0, 0.0, 0),
                    (4, 0, 4, 0.00244140625, 5.0, 0.0, 0),
                    (0, 1, 0, 25.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.5625, 2.0, 0.0, 0),
                    (2, 1, 2, 0.13020833333333334, 3.0, 0.0, 0),
                    (3, 1, 3, 0.01220703125, 4.0, 0.0, 0),
                    (4, 1, 4, 0.001220703125, 5.0, 0.0, 0),
                    (0, 2, 0, 12.5, 1.0, 0.0, 0),
                    (1, 2, 1, 0.78125, 2.0, 0.0, 0),
                    (2, 2, 2, 0.06510416666666667, 3.0, 0.0, 0),
                    (3, 2, 3, 0.006103515625, 4.0, 0.0, 0),
                    (4, 2, 4, 0.0006103515625, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )

    def test_auto_seq(self):
        target_hold_value = pd.DataFrame(
            {"a": [0.0, 70.0, 30.0, 0.0, 70.0], "b": [30.0, 0.0, 70.0, 30.0, 30.0], "c": [70.0, 30.0, 0.0, 70.0, 0.0]},
            index=price.index,
        )
        pd.testing.assert_frame_equal(
            from_orders_both(
                close=1.0,
                size=target_hold_value,
                size_type="targetvalue",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                call_seq="auto",
            ).get_asset_value(group_by=False),
            target_hold_value,
        )
        pd.testing.assert_frame_equal(
            from_orders_both(
                close=1.0,
                size=target_hold_value / 100,
                size_type="targetpercent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                call_seq="auto",
            ).get_asset_value(group_by=False),
            target_hold_value,
        )

    def test_max_orders(self):
        assert from_orders_both(close=price_wide).order_records.shape[0] == 9
        assert from_orders_both(close=price_wide, max_orders=3).order_records.shape[0] == 9
        assert from_orders_both(close=price_wide, max_orders=0).order_records.shape[0] == 0
        with pytest.raises(Exception):
            from_orders_both(close=price_wide, max_orders=2)

    def test_max_logs(self):
        assert from_orders_both(close=price_wide, log=True).log_records.shape[0] == 15
        assert from_orders_both(close=price_wide, log=True, max_logs=5).log_records.shape[0] == 15
        assert from_orders_both(close=price_wide, log=True, max_logs=0).log_records.shape[0] == 0
        with pytest.raises(Exception):
            from_orders_both(close=price_wide, log=True, max_logs=4)

    def test_jitted_parallel(self):
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        pf = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200, 300],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            log=True,
            jitted=dict(parallel=True),
        )
        pf2 = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200, 300],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            log=True,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            jitted=dict(parallel=True),
        )
        pf2 = from_orders_both(
            close=price_wide2,
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
        pf = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200, 300],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            log=True,
            chunked=True,
        )
        pf2 = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200, 300],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            log=True,
            chunked=False,
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            chunked=True,
        )
        pf2 = from_orders_both(
            close=price_wide2,
            init_cash=[100, 200],
            size=[1, 2, 3],
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            log=True,
            chunked=False,
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)

    def test_init_position(self):
        pf = vbt.Portfolio.from_orders(close=1, init_cash=0.0, init_position=1.0, size=-np.inf, direction="longonly")
        assert pf.init_position == 1.0
        assert_records_close(pf.order_records, np.array([(0, 0, 0, 1.0, 1.0, 0.0, 1)], dtype=order_dt))

    def test_cash_earnings(self):
        pf = vbt.Portfolio.from_orders(1, cash_earnings=[0, 1, 2, 3])
        pd.testing.assert_series_equal(pf.cash_earnings, pd.Series([0.0, 1.0, 2.0, 3.0]))
        assert_records_close(
            pf.order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 2, 1.0, 1.0, 0.0, 0), (2, 0, 3, 2.0, 1.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )

    def test_cash_dividends(self):
        pf = vbt.Portfolio.from_orders(1, size=np.inf, cash_dividends=[0, 1, 2, 3])
        pd.testing.assert_series_equal(pf.cash_earnings, pd.Series([0.0, 100.0, 400.0, 1800.0]))
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
        pd.testing.assert_frame_equal(
            from_orders_both(
                close=price_wide,
                fill_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_orders_both(
                close=price_wide,
                fill_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )
        pd.testing.assert_frame_equal(
            from_orders_longonly(
                close=price_wide,
                fill_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_orders_longonly(
                close=price_wide,
                fill_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )
        pd.testing.assert_frame_equal(
            from_orders_shortonly(
                close=price_wide,
                fill_returns=True,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
            from_orders_shortonly(
                close=price_wide,
                fill_returns=False,
                group_by=test_group_by,
                cash_sharing=test_cash_sharing,
            ).returns,
        )


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
        pd.testing.assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        pd.testing.assert_index_equal(pf.wrapper.columns, pd.Int64Index([0], dtype="int64"))
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
        pd.testing.assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        pd.testing.assert_index_equal(pf.wrapper.columns, pd.Index(["a", "b", "c"], dtype="object"))
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
        pd.testing.assert_index_equal(pf.wrapper.grouper.group_by, pd.Int64Index([0, 0, 1], dtype="int64"))
        pd.testing.assert_series_equal(
            pf.init_cash,
            pd.Series([200.0, 100.0], index=pd.Int64Index([0, 1], dtype="int64")).rename("init_cash"),
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
        pd.testing.assert_index_equal(pf.wrapper.grouper.group_by, pd.Int64Index([0, 0, 1], dtype="int64"))
        pd.testing.assert_series_equal(
            pf.init_cash,
            pd.Series([100.0, 100.0], index=pd.Int64Index([0, 1], dtype="int64")).rename("init_cash"),
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

    def test_sl_stop(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        with pytest.raises(Exception):
            from_signals_both(sl_stop=-0.1)

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

    def test_ts_stop(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        with pytest.raises(Exception):
            from_signals_both(sl_stop=-0.1, sl_trail=True)

        close = pd.Series([4.0, 5.0, 4.0, 3.0, 2.0], index=price.index)
        open = close + 0.25
        high = close + 0.5
        low = close - 0.5
        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
                sl_trail=True,
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
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
                sl_trail=True,
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
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
                sl_trail=True,
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
                sl_trail=True,
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
                sl_trail=True,
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.1, 0.5, np.inf]],
                sl_trail=True,
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
                sl_stop=[[np.nan, 0.15, 0.2, 0.25, np.inf]],
                sl_trail=True,
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
                sl_stop=[[np.nan, 0.15, 0.2, 0.25, np.inf]],
                sl_trail=True,
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
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
                sl_trail=True,
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
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
                sl_trail=True,
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
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
                sl_trail=True,
            ).order_records,
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
                sl_trail=True,
            ).order_records,
        )
        assert_records_close(
            from_signals_both(
                close=close,
                entries=exits,
                exits=entries,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
                sl_trail=True,
            ).order_records,
            from_signals_shortonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=[[np.nan, 0.5, 3.0, np.inf]],
                sl_trail=True,
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
                sl_trail=True,
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
                sl_stop=[[np.nan, 0.5, 0.75, 1.0, np.inf]],
                sl_trail=True,
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

    def test_tp_stop(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)

        with pytest.raises(Exception):
            from_signals_both(sl_stop=-0.1)

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

    def test_adjust_sl_func(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0], index=price.index)

        @njit
        def adjust_sl_func_nb(c, dur):
            return 0.0 if c.i - c.init_i >= dur else c.curr_stop, c.curr_trail

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=np.inf,
                adjust_sl_func_nb=adjust_sl_func_nb,
                adjust_sl_args=(2,),
            ).order_records,
            np.array([(0, 0, 0, 20.0, 5.0, 0.0, 0), (1, 0, 2, 20.0, 5.0, 0.0, 1)], dtype=order_dt),
        )

    def test_adjust_ts_func(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([10.0, 11.0, 12.0, 11.0, 10.0], index=price.index)

        @njit
        def adjust_sl_func_nb(c, dur):
            return 0.0 if c.i - c.curr_i >= dur else c.curr_stop, c.curr_trail

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                sl_stop=np.inf,
                adjust_sl_func_nb=adjust_sl_func_nb,
                adjust_sl_args=(2,),
            ).order_records,
            np.array([(0, 0, 0, 10.0, 10.0, 0.0, 0), (1, 0, 4, 10.0, 10.0, 0.0, 1)], dtype=order_dt),
        )

    def test_adjust_tp_func(self):
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=price.index)

        @njit
        def adjust_tp_func_nb(c, dur):
            return 0.0 if c.i - c.init_i >= dur else c.curr_stop

        assert_records_close(
            from_signals_longonly(
                close=close,
                entries=entries,
                exits=exits,
                tp_stop=np.inf,
                adjust_tp_func_nb=adjust_tp_func_nb,
                adjust_tp_args=(2,),
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
        pd.testing.assert_series_equal(pf.cash_earnings, pd.Series([0.0, 1.0, 2.0, 3.0]))
        assert_records_close(
            pf.order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 2, 1.0, 1.0, 0.0, 0), (2, 0, 3, 2.0, 1.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )

    def test_cash_dividends(self):
        pf = vbt.Portfolio.from_signals(1, size=np.inf, cash_dividends=[0, 1, 2, 3], accumulate=True)
        pd.testing.assert_series_equal(pf.cash_earnings, pd.Series([0.0, 100.0, 400.0, 1800.0]))
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
        pd.testing.assert_frame_equal(
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
        pd.testing.assert_frame_equal(
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
        pd.testing.assert_frame_equal(
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
            vbt.Portfolio.from_holding(price, sell_at_end=True).order_records,
            vbt.Portfolio.from_signals(price, entries, exits, accumulate=False).order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_holding(price, base_method="from_signals").order_records,
            vbt.Portfolio.from_holding(price, base_method="from_orders").order_records,
        )
        assert_records_close(
            vbt.Portfolio.from_holding(price, base_method="from_signals", sell_at_end=True).order_records,
            vbt.Portfolio.from_holding(price, base_method="from_orders", sell_at_end=True).order_records,
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
        pd.testing.assert_index_equal(result.wrapper.index, price.vbt.wrapper.index)
        pd.testing.assert_index_equal(result.wrapper.columns, price.vbt.wrapper.columns)
        result = vbt.Portfolio.from_random_signals(price, n=[1, 2], seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [[False, False], [False, True], [False, False], [True, True], [False, False]],
                [[False, False], [False, False], [False, True], [False, False], [True, True]],
            ).order_records,
        )
        pd.testing.assert_index_equal(
            result.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        pd.testing.assert_index_equal(result.wrapper.columns, pd.Int64Index([1, 2], dtype="int64", name="randnx_n"))

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
        pd.testing.assert_index_equal(result.wrapper.index, price.vbt.wrapper.index)
        pd.testing.assert_index_equal(result.wrapper.columns, price.vbt.wrapper.columns)
        result = vbt.Portfolio.from_random_signals(price, prob=[0.25, 0.5], seed=seed)
        assert_records_close(
            result.order_records,
            vbt.Portfolio.from_signals(
                price,
                [[False, True], [False, False], [False, False], [False, False], [True, False]],
                [[False, False], [False, True], [False, False], [False, False], [False, False]],
            ).order_records,
        )
        pd.testing.assert_index_equal(
            result.wrapper.index,
            pd.DatetimeIndex(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                dtype="datetime64[ns]",
                freq=None,
            ),
        )
        pd.testing.assert_index_equal(
            result.wrapper.columns,
            pd.MultiIndex.from_tuples([(0.25, 0.25), (0.5, 0.5)], names=["rprobnx_entry_prob", "rprobnx_exit_prob"]),
        )


# ############# from_order_func ############# #


@njit
def order_func_nb(c, size):
    _size = nb.get_elem_nb(c, size)
    return nb.order_nb(_size if c.i % 2 == 0 else -_size)


@njit
def log_order_func_nb(c, size):
    _size = nb.get_elem_nb(c, size)
    return nb.order_nb(_size if c.i % 2 == 0 else -_size, log=True)


@njit
def flex_order_func_nb(c, size):
    if c.call_idx < c.group_len:
        _size = nb.get_col_elem_nb(c, c.from_col + c.call_idx, size)
        return c.from_col + c.call_idx, nb.order_nb(_size if c.i % 2 == 0 else -_size)
    return -1, nb.order_nothing_nb()


@njit
def log_flex_order_func_nb(c, size):
    if c.call_idx < c.group_len:
        _size = nb.get_col_elem_nb(c, c.from_col + c.call_idx, size)
        return c.from_col + c.call_idx, nb.order_nb(_size if c.i % 2 == 0 else -_size, log=True)
    return -1, nb.order_nothing_nb()


class InOutputs(NamedTuple):
    custom_1d_arr: vbt.RepEval
    custom_2d_arr: vbt.RepEval
    custom_rec_arr: vbt.RepEval


class TestFromOrderFunc:
    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_one_column(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = vbt.Portfolio.from_order_func(
            price.tolist(),
            order_func,
            np.asarray(np.inf),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pf = vbt.Portfolio.from_order_func(
            price,
            order_func,
            np.asarray(np.inf),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pd.testing.assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        pd.testing.assert_index_equal(pf.wrapper.columns, pd.Int64Index([0], dtype="int64"))
        assert pf.wrapper.ndim == 1
        assert pf.wrapper.freq == day_dt
        assert pf.wrapper.grouper.group_by is None

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    @pytest.mark.parametrize("test_jitted", [False, True])
    def test_multiple_columns(self, test_row_wise, test_flexible, test_jitted):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            row_wise=test_row_wise,
            flexible=test_flexible,
            jitted=test_jitted,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 1.0, 1.0, 0.0, 0),
                    (1, 1, 1, 1.0, 2.0, 0.0, 1),
                    (2, 1, 2, 1.0, 3.0, 0.0, 0),
                    (3, 1, 3, 1.0, 4.0, 0.0, 1),
                    (4, 1, 4, 1.0, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pd.testing.assert_index_equal(
            pf.wrapper.index,
            pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]),
        )
        pd.testing.assert_index_equal(pf.wrapper.columns, pd.Index(["a", "b", "c"], dtype="object"))
        assert pf.wrapper.ndim == 2
        assert pf.wrapper.freq == day_dt
        assert pf.wrapper.grouper.group_by is None

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_price_area(self, test_row_wise, test_flexible):
        @njit
        def order_func2_nb(c, price, price_area_vio_mode):
            _price = nb.get_elem_nb(c, price)
            _price_area_vio_mode = nb.get_elem_nb(c, price_area_vio_mode)
            return nb.order_nb(
                1 if c.i % 2 == 0 else -1,
                _price,
                slippage=0.1,
                price_area_vio_mode=_price_area_vio_mode,
            )

        @njit
        def flex_order_func2_nb(c, price, price_area_vio_mode):
            if c.call_idx < c.group_len:
                _price = nb.get_col_elem_nb(c, c.from_col + c.call_idx, price)
                _price_area_vio_mode = nb.get_col_elem_nb(c, c.from_col + c.call_idx, price_area_vio_mode)
                return c.from_col + c.call_idx, nb.order_nb(
                    1 if c.i % 2 == 0 else -1,
                    _price,
                    slippage=0.1,
                    price_area_vio_mode=_price_area_vio_mode,
                )
            return -1, nb.order_nothing_nb()

        order_func = flex_order_func2_nb if test_flexible else order_func2_nb
        assert_records_close(
            vbt.Portfolio.from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(price=[[0.5, np.inf, 5]], price_area_vio_mode=PriceAreaVioMode.Ignore),
            ).order_records,
            np.array(
                [(0, 0, 0, 1.0, 0.55, 0.0, 0), (0, 1, 0, 1.0, 3.3, 0.0, 0), (0, 2, 0, 1.0, 5.5, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            vbt.Portfolio.from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(price=[[0.5, np.inf, 5]], price_area_vio_mode=PriceAreaVioMode.Cap),
            ).order_records,
            np.array(
                [(0, 0, 0, 1.0, 1.0, 0.0, 0), (0, 1, 0, 1.0, 3.0, 0.0, 0), (0, 2, 0, 1.0, 4.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )
        with pytest.raises(Exception):
            vbt.Portfolio.from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(price=0.5, price_area_vio_mode=PriceAreaVioMode.Error),
            )
        with pytest.raises(Exception):
            vbt.Portfolio.from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(price=np.inf, price_area_vio_mode=PriceAreaVioMode.Error),
            )
        with pytest.raises(Exception):
            vbt.Portfolio.from_order_func(
                3,
                order_func,
                vbt.Rep("price"),
                vbt.Rep("price_area_vio_mode"),
                open=2,
                high=4,
                low=1,
                row_wise=test_row_wise,
                flexible=test_flexible,
                broadcast_named_args=dict(price=5, price_area_vio_mode=PriceAreaVioMode.Error),
            )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_group_by(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            np.asarray(np.inf),
            group_by=np.array([0, 0, 1]),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 1, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 1, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pd.testing.assert_index_equal(pf.wrapper.grouper.group_by, pd.Int64Index([0, 0, 1], dtype="int64"))
        pd.testing.assert_series_equal(
            pf.init_cash,
            pd.Series([200.0, 100.0], index=pd.Int64Index([0, 1], dtype="int64")).rename("init_cash"),
        )
        assert not pf.cash_sharing

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_cash_sharing(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            np.asarray(np.inf),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        pd.testing.assert_index_equal(pf.wrapper.grouper.group_by, pd.Int64Index([0, 0, 1], dtype="int64"))
        pd.testing.assert_series_equal(
            pf.init_cash,
            pd.Series([100.0, 100.0], index=pd.Int64Index([0, 1], dtype="int64")).rename("init_cash"),
        )
        assert pf.cash_sharing

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_call_seq(self, test_row_wise):
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            np.asarray(np.inf),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            row_wise=test_row_wise,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 100.0, 1.0, 0.0, 0),
                    (1, 0, 1, 200.0, 2.0, 0.0, 1),
                    (2, 0, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 0, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 0, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]),
        )
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            np.asarray(np.inf),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            call_seq="reversed",
            row_wise=test_row_wise,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 1, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 1, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            np.asarray(np.inf),
            group_by=np.array([0, 0, 1]),
            cash_sharing=True,
            call_seq="random",
            seed=seed,
            row_wise=test_row_wise,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 1, 0, 100.0, 1.0, 0.0, 0),
                    (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 1, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 1, 4, 53.33333333333335, 5.0, 0.0, 0),
                    (0, 2, 0, 100.0, 1.0, 0.0, 0),
                    (1, 2, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (3, 2, 3, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 2, 4, 53.33333333333335, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
        )
        with pytest.raises(Exception):
            vbt.Portfolio.from_order_func(
                price_wide,
                order_func_nb,
                np.asarray(np.inf),
                group_by=np.array([0, 0, 1]),
                cash_sharing=True,
                call_seq="auto",
                row_wise=test_row_wise,
            )

        target_hold_value = pd.DataFrame(
            {"a": [0.0, 70.0, 30.0, 0.0, 70.0], "b": [30.0, 0.0, 70.0, 30.0, 30.0], "c": [70.0, 30.0, 0.0, 70.0, 0.0]},
            index=price.index,
        )

        @njit
        def pre_segment_func_nb(c, target_hold_value):
            order_size = np.copy(target_hold_value[c.i, c.from_col : c.to_col])
            order_size_type = np.full(c.group_len, SizeType.TargetValue)
            direction = np.full(c.group_len, Direction.Both)
            order_value_out = np.empty(c.group_len, dtype=np.float_)
            c.last_val_price[c.from_col : c.to_col] = c.close[c.i, c.from_col : c.to_col]
            nb.sort_call_seq_nb(c, order_size, order_size_type, direction, order_value_out)
            return order_size, order_size_type, direction

        @njit
        def pct_order_func_nb(c, order_size, order_size_type, direction):
            col_i = c.call_seq_now[c.call_idx]
            return nb.order_nb(
                order_size[col_i],
                c.close[c.i, col_i],
                size_type=order_size_type[col_i],
                direction=direction[col_i],
            )

        pf = vbt.Portfolio.from_order_func(
            price_wide * 0 + 1,
            pct_order_func_nb,
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(target_hold_value.values,),
            row_wise=test_row_wise,
        )
        np.testing.assert_array_equal(
            pf.call_seq.values,
            np.array([[0, 1, 2], [2, 1, 0], [0, 2, 1], [1, 0, 2], [2, 1, 0]]),
        )
        pd.testing.assert_frame_equal(pf.get_asset_value(group_by=False), target_hold_value)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_target_value(self, test_row_wise, test_flexible):
        @njit
        def target_val_pre_segment_func_nb(c, val_price):
            c.last_val_price[c.from_col : c.to_col] = val_price[c.i]
            return ()

        if test_flexible:

            @njit
            def target_val_order_func_nb(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(50.0, nb.get_col_elem_nb(c, col, c.close), size_type=SizeType.TargetValue)
                return -1, nb.order_nothing_nb()

        else:

            @njit
            def target_val_order_func_nb(c):
                return nb.order_nb(50.0, nb.get_elem_nb(c, c.close), size_type=SizeType.TargetValue)

        pf = vbt.Portfolio.from_order_func(
            price.iloc[1:],
            target_val_order_func_nb,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 1, 25.0, 3.0, 0.0, 0),
                    (1, 0, 2, 8.333333333333332, 4.0, 0.0, 1),
                    (2, 0, 3, 4.166666666666668, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = vbt.Portfolio.from_order_func(
            price.iloc[1:],
            target_val_order_func_nb,
            pre_segment_func_nb=target_val_pre_segment_func_nb,
            pre_segment_args=(price.iloc[:-1].values,),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 50.0, 2.0, 0.0, 0),
                    (1, 0, 1, 25.0, 3.0, 0.0, 1),
                    (2, 0, 2, 8.333333333333332, 4.0, 0.0, 1),
                    (3, 0, 3, 4.166666666666668, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_target_percent(self, test_row_wise, test_flexible):
        @njit
        def target_pct_pre_segment_func_nb(c, val_price):
            c.last_val_price[c.from_col : c.to_col] = val_price[c.i]
            return ()

        if test_flexible:

            @njit
            def target_pct_order_func_nb(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(0.5, nb.get_col_elem_nb(c, col, c.close), size_type=SizeType.TargetPercent)
                return -1, nb.order_nothing_nb()

        else:

            @njit
            def target_pct_order_func_nb(c):
                return nb.order_nb(0.5, nb.get_elem_nb(c, c.close), size_type=SizeType.TargetPercent)

        pf = vbt.Portfolio.from_order_func(
            price.iloc[1:],
            target_pct_order_func_nb,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 1, 25.0, 3.0, 0.0, 0),
                    (1, 0, 2, 8.333333333333332, 4.0, 0.0, 1),
                    (2, 0, 3, 1.0416666666666679, 5.0, 0.0, 1),
                ],
                dtype=order_dt,
            ),
        )
        pf = vbt.Portfolio.from_order_func(
            price.iloc[1:],
            target_pct_order_func_nb,
            pre_segment_func_nb=target_pct_pre_segment_func_nb,
            pre_segment_args=(price.iloc[:-1].values,),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [(0, 0, 0, 50.0, 2.0, 0.0, 0), (1, 0, 1, 25.0, 3.0, 0.0, 1), (2, 0, 3, 3.125, 5.0, 0.0, 1)],
                dtype=order_dt,
            ),
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_update_value(self, test_row_wise, test_flexible):
        if test_flexible:

            @njit
            def order_func_nb(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(
                        np.inf if c.i % 2 == 0 else -np.inf,
                        nb.get_col_elem_nb(c, col, c.close),
                        fees=0.01,
                        fixed_fees=1.0,
                        slippage=0.01,
                    )
                return -1, nb.order_nothing_nb()

        else:

            @njit
            def order_func_nb(c):
                return nb.order_nb(
                    np.inf if c.i % 2 == 0 else -np.inf,
                    nb.get_elem_nb(c, c.close),
                    fees=0.01,
                    fixed_fees=1.0,
                    slippage=0.01,
                )

        @njit
        def post_order_func_nb(c, value_before, value_now):
            value_before[c.i, c.col] = c.value_before
            value_now[c.i, c.col] = c.value_now

        value_before = np.empty_like(price.values[:, None])
        value_now = np.empty_like(price.values[:, None])

        vbt.Portfolio.from_order_func(
            price,
            order_func_nb,
            post_order_func_nb=post_order_func_nb,
            post_order_args=(value_before, value_now),
            row_wise=test_row_wise,
            update_value=False,
            flexible=test_flexible,
        )

        np.testing.assert_array_equal(value_before, value_now)

        vbt.Portfolio.from_order_func(
            price,
            order_func_nb,
            post_order_func_nb=post_order_func_nb,
            post_order_args=(value_before, value_now),
            row_wise=test_row_wise,
            update_value=True,
            flexible=test_flexible,
        )

        np.testing.assert_array_equal(
            value_before,
            np.array([[100.0], [97.04930889128518], [185.46988117104038], [82.47853456223025], [104.65775576218027]]),
        )
        np.testing.assert_array_equal(
            value_now,
            np.array(
                [
                    [98.01980198019803],
                    [187.36243097890815],
                    [83.30331990785257],
                    [105.72569204546781],
                    [73.54075125567473],
                ]
            ),
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_states(self, test_row_wise, test_flexible):
        cash_deposits = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [100.0, 0.0], [0.0, 0.0]])
        cash_earnings = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        close = np.array([[1, 1, 1], [np.nan, 2, 2], [3, np.nan, 3], [4, 4, np.nan], [5, 5, 5]])
        open = close - 0.1
        size = np.array([[1, 1, 1], [-1, -1, -1], [1, 1, 1], [-1, -1, -1], [1, 1, 1]])
        value_arr1 = np.empty((size.shape[0], 2), dtype=np.float_)
        value_arr2 = np.empty(size.shape, dtype=np.float_)
        value_arr3 = np.empty(size.shape, dtype=np.float_)
        return_arr1 = np.empty((size.shape[0], 2), dtype=np.float_)
        return_arr2 = np.empty(size.shape, dtype=np.float_)
        return_arr3 = np.empty(size.shape, dtype=np.float_)
        pos_record_arr1 = np.empty(size.shape, dtype=trade_dt)
        pos_record_arr2 = np.empty(size.shape, dtype=trade_dt)
        pos_record_arr3 = np.empty(size.shape, dtype=trade_dt)

        def pre_segment_func_nb(c):
            value_arr1[c.i, c.group] = c.last_value[c.group]
            return_arr1[c.i, c.group] = c.last_return[c.group]
            for col in range(c.from_col, c.to_col):
                pos_record_arr1[c.i, col] = c.last_pos_record[col]
            c.last_val_price[c.from_col : c.to_col] = c.last_val_price[c.from_col : c.to_col] + 0.5
            return ()

        if test_flexible:

            def order_func_nb(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    value_arr2[c.i, col] = c.last_value[c.group]
                    return_arr2[c.i, col] = c.last_return[c.group]
                    pos_record_arr2[c.i, col] = c.last_pos_record[col]
                    return col, nb.order_nb(size[c.i, col], fixed_fees=1.0)
                return -1, nb.order_nothing_nb()

        else:

            def order_func_nb(c):
                value_arr2[c.i, c.col] = c.value_now
                return_arr2[c.i, c.col] = c.return_now
                pos_record_arr2[c.i, c.col] = c.pos_record_now
                return nb.order_nb(size[c.i, c.col], fixed_fees=1.0)

        def post_order_func_nb(c):
            value_arr3[c.i, c.col] = c.value_now
            return_arr3[c.i, c.col] = c.return_now
            pos_record_arr3[c.i, c.col] = c.pos_record_now

        vbt.Portfolio.from_order_func(
            close,
            order_func_nb,
            pre_segment_func_nb=pre_segment_func_nb,
            post_order_func_nb=post_order_func_nb,
            jitted=False,
            open=open,
            update_value=True,
            ffill_val_price=True,
            group_by=[0, 0, 1],
            cash_sharing=True,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )

        np.testing.assert_array_equal(
            value_arr1,
            np.array([[100.0, 100.0], [98.9, 99.9], [99.9, 99.0], [100.8, 98.0], [200.0, 99.9]]),
        )
        np.testing.assert_array_equal(
            value_arr2,
            np.array(
                [
                    [100.0, 99.0, 100.0],
                    [99.9, 99.9, 100.4],
                    [100.4, 99.0, 99.0],
                    [201.8, 200.0, 98.5],
                    [200.0, 198.6, 100.4],
                ]
            ),
        )
        np.testing.assert_array_equal(
            value_arr3,
            np.array(
                [
                    [99.0, 98.0, 99.0],
                    [99.9, 98.5, 99.0],
                    [99.0, 99.0, 98.0],
                    [200.0, 199.0, 98.5],
                    [198.6, 198.0, 99.0],
                ],
            ),
        )
        np.testing.assert_array_equal(
            return_arr1,
            np.array(
                [
                    [0.0, 0.0],
                    [0.009183673469387813, 0.009090909090909148],
                    [0.014213197969543205, 0.0],
                    [0.018181818181818153, 0.0],
                    [0.0, 0.014213197969543205],
                ]
            ),
        )
        np.testing.assert_array_equal(
            return_arr2,
            np.array(
                [
                    [0.0, -0.01, 0.0],
                    [0.019387755102040875, 0.019387755102040875, 0.0141414141414142],
                    [0.0192893401015229, 0.005076142131979695, 0.0],
                    [0.0282828282828284, 0.010101010101010102, 0.00510204081632653],
                    [0.0, -0.007000000000000029, 0.0192893401015229],
                ]
            ),
        )
        np.testing.assert_array_equal(
            return_arr3,
            np.array(
                [
                    [-0.01, -0.02, -0.01],
                    [0.019387755102040875, 0.00510204081632653, 0.0],
                    [0.005076142131979695, 0.005076142131979695, -0.010101010101010102],
                    [0.010101010101010102, 0.0, 0.00510204081632653],
                    [-0.007000000000000029, -0.01, 0.005076142131979695],
                ]
            ),
        )
        assert_records_close(
            pos_record_arr1.flatten()[3:],
            np.array(
                [
                    (0, 0, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, -1.0, -1.0, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, -0.10000000000000009, -0.10000000000000009, 0, 0, 0),
                    (0, 2, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, -0.10000000000000009, -0.10000000000000009, 0, 0, 0),
                    (0, 0, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, 0.8999999999999999, 0.8999999999999999, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 2, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 0, 2.0, 0, 2.0, 2.0, -1, np.nan, 0.0, 1.7999999999999998, 0.44999999999999996, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (1, 2, 1.0, 2, 3.0, 1.0, -1, np.nan, 0.0, -1.0, -0.3333333333333333, 0, 0, 1),
                    (0, 0, 2.0, 0, 2.0, 2.0, -1, 4.0, 1.0, 1.9000000000000004, 0.4750000000000001, 0, 0, 0),
                    (1, 1, 1.0, 3, 4.0, 1.0, -1, np.nan, 0.0, -1.9000000000000004, -0.4750000000000001, 1, 0, 1),
                    (1, 2, 1.0, 2, 3.0, 1.0, -1, np.nan, 0.0, 0.9000000000000004, 0.3000000000000001, 0, 0, 1),
                ],
                dtype=trade_dt,
            ),
        )
        assert_records_close(
            pos_record_arr2.flatten()[3:],
            np.array(
                [
                    (0, 0, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, -0.5, -0.5, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, 0.3999999999999999, 0.3999999999999999, 0, 0, 0),
                    (0, 2, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, 0.3999999999999999, 0.3999999999999999, 0, 0, 0),
                    (0, 0, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, 1.4, 1.4, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 2, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 0, 2.0, 0, 2.0, 2.0, -1, np.nan, 0.0, 2.8000000000000007, 0.7000000000000002, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (1, 2, 1.0, 2, 3.0, 1.0, -1, np.nan, 0.0, -0.5, -0.16666666666666666, 0, 0, 1),
                    (0, 0, 2.0, 0, 2.0, 2.0, -1, 4.0, 1.0, 2.4000000000000004, 0.6000000000000001, 0, 0, 0),
                    (1, 1, 1.0, 3, 4.0, 1.0, -1, np.nan, 0.0, -2.4000000000000004, -0.6000000000000001, 1, 0, 1),
                    (1, 2, 1.0, 2, 3.0, 1.0, -1, np.nan, 0.0, 1.4000000000000004, 0.4666666666666668, 0, 0, 1),
                ],
                dtype=trade_dt,
            ),
        )
        assert_records_close(
            pos_record_arr3.flatten(),
            np.array(
                [
                    (0, 0, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, -1.0, -1.0, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, -1.0, -1.0, 0, 0, 0),
                    (0, 2, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, -1.0, -1.0, 0, 0, 0),
                    (0, 0, 1.0, 0, 1.0, 1.0, -1, np.nan, 0.0, -0.5, -0.5, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 2, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (0, 0, 2.0, 0, 2.0, 2.0, -1, np.nan, 0.0, 0.0, 0.0, 0, 0, 0),
                    (0, 1, 1.0, 0, 1.0, 1.0, 1, 2.0, 1.0, -1.0, -1.0, 0, 1, 0),
                    (1, 2, 1.0, 2, 3.0, 1.0, -1, np.nan, 0.0, -1.0, -0.3333333333333333, 0, 0, 1),
                    (0, 0, 2.0, 0, 2.0, 2.0, -1, 4.0, 1.0, 1.0, 0.25, 0, 0, 0),
                    (1, 1, 1.0, 3, 4.0, 1.0, -1, np.nan, 0.0, -1.0, -0.25, 1, 0, 1),
                    (1, 2, 1.0, 2, 3.0, 1.0, -1, np.nan, 0.0, -0.5, -0.16666666666666666, 0, 0, 1),
                    (0, 0, 3.0, 0, 3.0, 3.0, -1, 4.0, 1.0, 1.0, 0.1111111111111111, 0, 0, 0),
                    (1, 1, 1.0, 3, 4.0, 1.0, 4, 5.0, 1.0, -3.0, -0.75, 1, 1, 1),
                    (1, 2, 2.0, 2, 4.0, 2.0, -1, np.nan, 0.0, 0.0, 0.0, 0, 0, 1),
                ],
                dtype=trade_dt,
            ),
        )

        cash_arr = np.empty((size.shape[0], 2), dtype=np.float_)
        position_arr = np.empty(size.shape, dtype=np.float_)
        val_price_arr = np.empty(size.shape, dtype=np.float_)
        value_arr = np.empty((size.shape[0], 2), dtype=np.float_)
        return_arr = np.empty((size.shape[0], 2), dtype=np.float_)
        pos_record_arr = np.empty(size.shape[1], dtype=trade_dt)

        def post_segment_func_nb(c):
            cash_arr[c.i, c.group] = c.last_cash[c.group]
            for col in range(c.from_col, c.to_col):
                position_arr[c.i, col] = c.last_position[col]
                val_price_arr[c.i, col] = c.last_val_price[col]
            value_arr[c.i, c.group] = c.last_value[c.group]
            return_arr[c.i, c.group] = c.last_return[c.group]

        def post_sim_func_nb(c):
            pos_record_arr[:] = c.last_pos_record

        pf = vbt.Portfolio.from_order_func(
            close,
            order_func_nb,
            post_segment_func_nb=post_segment_func_nb,
            post_sim_func_nb=post_sim_func_nb,
            jitted=False,
            update_value=True,
            ffill_val_price=True,
            group_by=[0, 0, 1],
            cash_sharing=True,
            cash_deposits=cash_deposits,
            cash_earnings=cash_earnings,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )

        np.testing.assert_array_equal(cash_arr, pf.cash.values)
        np.testing.assert_array_equal(position_arr, pf.assets.values)
        np.testing.assert_array_equal(val_price_arr, pf.filled_close.values)
        np.testing.assert_array_equal(value_arr, pf.value.values)
        np.testing.assert_array_equal(return_arr, pf.returns.values)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_post_sim_ctx(self, test_row_wise, test_flexible):
        if test_flexible:

            def order_func(c):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(
                        1.0,
                        nb.get_col_elem_nb(c, col, c.close),
                        fees=0.01,
                        fixed_fees=1.0,
                        slippage=0.01,
                        log=True,
                    )
                return -1, nb.order_nothing_nb()

        else:

            def order_func(c):
                return nb.order_nb(1.0, nb.get_elem_nb(c, c.close), fees=0.01, fixed_fees=1.0, slippage=0.01, log=True)

        def post_sim_func(c, lst):
            lst.append(deepcopy(c))

        lst = []

        vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            post_sim_func_nb=post_sim_func,
            post_sim_args=(lst,),
            row_wise=test_row_wise,
            update_value=True,
            jitted=False,
            group_by=[0, 0, 1],
            cash_sharing=True,
            keep_inout_raw=False,
            max_logs=price_wide.shape[0],
            flexible=test_flexible,
        )

        c = lst[-1]

        assert c.target_shape == price_wide.shape
        np.testing.assert_array_equal(c.close, price_wide.values)
        np.testing.assert_array_equal(c.group_lens, np.array([2, 1]))
        assert c.cash_sharing
        if test_flexible:
            assert c.call_seq is None
        else:
            np.testing.assert_array_equal(c.call_seq, np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]))
        np.testing.assert_array_equal(c.init_cash, np.array([100.0, 100.0]))
        np.testing.assert_array_equal(c.init_position, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(
            c.segment_mask,
            np.array([[True, True], [True, True], [True, True], [True, True], [True, True]]),
        )
        assert c.ffill_val_price
        assert c.update_value
        assert_records_close(
            c.order_records.flatten(order="F"),
            np.array(
                [
                    (0, 0, 0, 1.0, 1.01, 1.0101, 0),
                    (1, 0, 1, 1.0, 2.02, 1.0202, 0),
                    (2, 0, 2, 1.0, 3.0300000000000002, 1.0303, 0),
                    (3, 0, 3, 1.0, 4.04, 1.0404, 0),
                    (4, 0, 4, 1.0, 5.05, 1.0505, 0),
                    (0, 1, 0, 1.0, 1.01, 1.0101, 0),
                    (1, 1, 1, 1.0, 2.02, 1.0202, 0),
                    (2, 1, 2, 1.0, 3.0300000000000002, 1.0303, 0),
                    (3, 1, 3, 1.0, 4.04, 1.0404, 0),
                    (4, 1, 4, 1.0, 5.05, 1.0505, 0),
                    (0, 2, 0, 1.0, 1.01, 1.0101, 0),
                    (1, 2, 1, 1.0, 2.02, 1.0202, 0),
                    (2, 2, 2, 1.0, 3.0300000000000002, 1.0303, 0),
                    (3, 2, 3, 1.0, 4.04, 1.0404, 0),
                    (4, 2, 4, 1.0, 5.05, 1.0505, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert_records_close(
            c.log_records.flatten(order="F"),
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
                        np.nan,
                        100.0,
                        1.0,
                        1.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        97.9799,
                        1.0,
                        0.0,
                        97.9799,
                        1.01,
                        98.9899,
                        1.0,
                        1.01,
                        1.0101,
                        0,
                        0,
                        -1,
                        0,
                    ),
                    (
                        1,
                        0,
                        0,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        95.9598,
                        1.0,
                        0.0,
                        95.9598,
                        1.0,
                        97.9598,
                        1.0,
                        2.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        92.9196,
                        2.0,
                        0.0,
                        92.9196,
                        2.02,
                        97.95960000000001,
                        1.0,
                        2.02,
                        1.0202,
                        0,
                        0,
                        -1,
                        1,
                    ),
                    (
                        2,
                        0,
                        0,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        89.8794,
                        2.0,
                        0.0,
                        89.8794,
                        2.0,
                        97.8794,
                        1.0,
                        3.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        85.8191,
                        3.0,
                        0.0,
                        85.8191,
                        3.0300000000000002,
                        98.90910000000001,
                        1.0,
                        3.0300000000000002,
                        1.0303,
                        0,
                        0,
                        -1,
                        2,
                    ),
                    (
                        3,
                        0,
                        0,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        81.75880000000001,
                        3.0,
                        0.0,
                        81.75880000000001,
                        3.0,
                        99.75880000000001,
                        1.0,
                        4.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        76.67840000000001,
                        4.0,
                        0.0,
                        76.67840000000001,
                        4.04,
                        101.83840000000001,
                        1.0,
                        4.04,
                        1.0404,
                        0,
                        0,
                        -1,
                        3,
                    ),
                    (
                        4,
                        0,
                        0,
                        4,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        71.59800000000001,
                        4.0,
                        0.0,
                        71.59800000000001,
                        4.0,
                        103.59800000000001,
                        1.0,
                        5.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        65.49750000000002,
                        5.0,
                        0.0,
                        65.49750000000002,
                        5.05,
                        106.74750000000002,
                        1.0,
                        5.05,
                        1.0505,
                        0,
                        0,
                        -1,
                        4,
                    ),
                    (
                        0,
                        0,
                        1,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        97.9799,
                        0.0,
                        0.0,
                        97.9799,
                        np.nan,
                        98.9899,
                        1.0,
                        1.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        95.9598,
                        1.0,
                        0.0,
                        95.9598,
                        1.01,
                        97.97980000000001,
                        1.0,
                        1.01,
                        1.0101,
                        0,
                        0,
                        -1,
                        0,
                    ),
                    (
                        1,
                        0,
                        1,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        92.9196,
                        1.0,
                        0.0,
                        92.9196,
                        1.0,
                        97.95960000000001,
                        1.0,
                        2.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        89.8794,
                        2.0,
                        0.0,
                        89.8794,
                        2.02,
                        97.95940000000002,
                        1.0,
                        2.02,
                        1.0202,
                        0,
                        0,
                        -1,
                        1,
                    ),
                    (
                        2,
                        0,
                        1,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        85.8191,
                        2.0,
                        0.0,
                        85.8191,
                        2.0,
                        98.90910000000001,
                        1.0,
                        3.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        81.75880000000001,
                        3.0,
                        0.0,
                        81.75880000000001,
                        3.0300000000000002,
                        99.93880000000001,
                        1.0,
                        3.0300000000000002,
                        1.0303,
                        0,
                        0,
                        -1,
                        2,
                    ),
                    (
                        3,
                        0,
                        1,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        76.67840000000001,
                        3.0,
                        0.0,
                        76.67840000000001,
                        3.0,
                        101.83840000000001,
                        1.0,
                        4.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        71.59800000000001,
                        4.0,
                        0.0,
                        71.59800000000001,
                        4.04,
                        103.918,
                        1.0,
                        4.04,
                        1.0404,
                        0,
                        0,
                        -1,
                        3,
                    ),
                    (
                        4,
                        0,
                        1,
                        4,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        65.49750000000002,
                        4.0,
                        0.0,
                        65.49750000000002,
                        4.0,
                        106.74750000000002,
                        1.0,
                        5.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        59.39700000000002,
                        5.0,
                        0.0,
                        59.39700000000002,
                        5.05,
                        109.89700000000002,
                        1.0,
                        5.05,
                        1.0505,
                        0,
                        0,
                        -1,
                        4,
                    ),
                    (
                        0,
                        1,
                        2,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        1.0,
                        100.0,
                        0.0,
                        0.0,
                        100.0,
                        np.nan,
                        100.0,
                        1.0,
                        1.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        97.9799,
                        1.0,
                        0.0,
                        97.9799,
                        1.01,
                        98.9899,
                        1.0,
                        1.01,
                        1.0101,
                        0,
                        0,
                        -1,
                        0,
                    ),
                    (
                        1,
                        1,
                        2,
                        1,
                        np.nan,
                        np.nan,
                        np.nan,
                        2.0,
                        97.9799,
                        1.0,
                        0.0,
                        97.9799,
                        1.0,
                        98.9799,
                        1.0,
                        2.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        94.9397,
                        2.0,
                        0.0,
                        94.9397,
                        2.02,
                        98.97970000000001,
                        1.0,
                        2.02,
                        1.0202,
                        0,
                        0,
                        -1,
                        1,
                    ),
                    (
                        2,
                        1,
                        2,
                        2,
                        np.nan,
                        np.nan,
                        np.nan,
                        3.0,
                        94.9397,
                        2.0,
                        0.0,
                        94.9397,
                        2.0,
                        98.9397,
                        1.0,
                        3.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        90.8794,
                        3.0,
                        0.0,
                        90.8794,
                        3.0300000000000002,
                        99.96940000000001,
                        1.0,
                        3.0300000000000002,
                        1.0303,
                        0,
                        0,
                        -1,
                        2,
                    ),
                    (
                        3,
                        1,
                        2,
                        3,
                        np.nan,
                        np.nan,
                        np.nan,
                        4.0,
                        90.8794,
                        3.0,
                        0.0,
                        90.8794,
                        3.0,
                        99.8794,
                        1.0,
                        4.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        85.799,
                        4.0,
                        0.0,
                        85.799,
                        4.04,
                        101.959,
                        1.0,
                        4.04,
                        1.0404,
                        0,
                        0,
                        -1,
                        3,
                    ),
                    (
                        4,
                        1,
                        2,
                        4,
                        np.nan,
                        np.nan,
                        np.nan,
                        5.0,
                        85.799,
                        4.0,
                        0.0,
                        85.799,
                        4.0,
                        101.799,
                        1.0,
                        5.0,
                        0,
                        2,
                        0.01,
                        1.0,
                        0.01,
                        0.0,
                        np.inf,
                        np.nan,
                        0.0,
                        0,
                        False,
                        True,
                        False,
                        True,
                        79.69850000000001,
                        5.0,
                        0.0,
                        79.69850000000001,
                        5.05,
                        104.94850000000001,
                        1.0,
                        5.05,
                        1.0505,
                        0,
                        0,
                        -1,
                        4,
                    ),
                ],
                dtype=log_dt,
            ),
        )
        np.testing.assert_array_equal(c.last_cash, np.array([59.39700000000002, 79.69850000000001]))
        np.testing.assert_array_equal(c.last_position, np.array([5.0, 5.0, 5.0]))
        np.testing.assert_array_equal(c.last_val_price, np.array([5.0, 5.0, 5.0]))
        np.testing.assert_array_equal(c.last_value, np.array([109.39700000000002, 104.69850000000001]))
        np.testing.assert_array_equal(c.last_return, np.array([0.05597598409235705, 0.028482598060884715]))
        np.testing.assert_array_equal(c.last_debt, np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(c.last_free_cash, np.array([59.39700000000002, 79.69850000000001]))
        np.testing.assert_array_equal(c.last_oidx, np.array([4, 4, 4]))
        np.testing.assert_array_equal(c.last_lidx, np.array([4, 4, 4]))

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_free_cash(self, test_row_wise, test_flexible):
        if test_flexible:

            def order_func(c, size):
                col = c.from_col + c.call_idx
                if c.call_idx < c.group_len:
                    return col, nb.order_nb(
                        size[c.i, col],
                        nb.get_col_elem_nb(c, col, c.close),
                        fees=0.01,
                        fixed_fees=1.0,
                        slippage=0.01,
                    )
                return -1, nb.order_nothing_nb()

        else:

            def order_func(c, size):
                return nb.order_nb(
                    size[c.i, c.col],
                    nb.get_elem_nb(c, c.close),
                    fees=0.01,
                    fixed_fees=1.0,
                    slippage=0.01,
                )

        def post_order_func(c, debt, free_cash):
            debt[c.i, c.col] = c.debt_now
            if c.cash_sharing:
                free_cash[c.i, c.group] = c.free_cash_now
            else:
                free_cash[c.i, c.col] = c.free_cash_now

        size = np.array([[5, -5, 5], [5, -5, -10], [-5, 5, 10], [-5, 5, -10], [-5, 5, 10]])
        debt = np.empty(price_wide.shape, dtype=np.float_)
        free_cash = np.empty(price_wide.shape, dtype=np.float_)
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            size,
            post_order_func_nb=post_order_func,
            post_order_args=(
                debt,
                free_cash,
            ),
            row_wise=test_row_wise,
            jitted=False,
            flexible=test_flexible,
        )
        np.testing.assert_array_equal(
            debt,
            np.array(
                [
                    [0.0, 4.95, 0.0],
                    [0.0, 14.850000000000001, 9.9],
                    [0.0, 7.425000000000001, 0.0],
                    [0.0, 0.0, 19.8],
                    [24.75, 0.0, 0.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            free_cash,
            np.array(
                [
                    [93.8995, 94.0005, 93.8995],
                    [82.6985, 83.00150000000001, 92.70150000000001],
                    [96.39999999999999, 81.55000000000001, 80.8985],
                    [115.002, 74.998, 79.5025],
                    [89.0045, 48.49550000000001, 67.0975],
                ]
            ),
        )
        np.testing.assert_almost_equal(free_cash, pf.get_cash(free=True).values)

        debt = np.empty(price_wide.shape, dtype=np.float_)
        free_cash = np.empty(price_wide.shape, dtype=np.float_)
        pf = vbt.Portfolio.from_order_func(
            price_wide.vbt.wrapper.wrap(price_wide.values[::-1]),
            order_func,
            size,
            post_order_func_nb=post_order_func,
            post_order_args=(
                debt,
                free_cash,
            ),
            row_wise=test_row_wise,
            jitted=False,
            flexible=test_flexible,
        )
        np.testing.assert_array_equal(
            debt,
            np.array([[0.0, 24.75, 0.0], [0.0, 44.55, 19.8], [0.0, 22.275, 0.0], [0.0, 0.0, 9.9], [4.95, 0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            free_cash,
            np.array(
                [
                    [73.4975, 74.0025, 73.4975],
                    [52.0955, 53.00449999999999, 72.1015],
                    [65.797, 81.25299999999999, 80.0985],
                    [74.598, 114.60199999999998, 78.9005],
                    [68.5985, 108.50149999999998, 87.49949999999998],
                ]
            ),
        )
        np.testing.assert_almost_equal(free_cash, pf.get_cash(free=True).values)

        debt = np.empty(price_wide.shape, dtype=np.float_)
        free_cash = np.empty((price_wide.shape[0], 2), dtype=np.float_)
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            size,
            post_order_func_nb=post_order_func,
            post_order_args=(
                debt,
                free_cash,
            ),
            row_wise=test_row_wise,
            jitted=False,
            group_by=[0, 0, 1],
            cash_sharing=True,
            flexible=test_flexible,
        )
        np.testing.assert_array_equal(
            debt,
            np.array(
                [
                    [0.0, 4.95, 0.0],
                    [0.0, 14.850000000000001, 9.9],
                    [0.0, 7.425000000000001, 0.0],
                    [0.0, 0.0, 19.8],
                    [24.75, 0.0, 0.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            free_cash,
            np.array(
                [
                    [87.9, 93.8995],
                    [65.70000000000002, 92.70150000000001],
                    [77.95000000000002, 80.8985],
                    [90.00000000000001, 79.5025],
                    [37.500000000000014, 67.0975],
                ]
            ),
        )
        np.testing.assert_almost_equal(free_cash, pf.get_cash(free=True).values)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_init_cash(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            np.asarray(10.0),
            row_wise=test_row_wise,
            init_cash=[1.0, 10.0, np.inf],
            flexible=test_flexible,
        )
        assert_records_close(
            pf.order_records,
            np.array(
                [
                    (0, 0, 0, 1.0, 1.0, 0.0, 0),
                    (1, 0, 1, 10.0, 2.0, 0.0, 1),
                    (2, 0, 2, 6.666666666666667, 3.0, 0.0, 0),
                    (3, 0, 3, 10.0, 4.0, 0.0, 1),
                    (4, 0, 4, 8.0, 5.0, 0.0, 0),
                    (0, 1, 0, 10.0, 1.0, 0.0, 0),
                    (1, 1, 1, 10.0, 2.0, 0.0, 1),
                    (2, 1, 2, 6.666666666666667, 3.0, 0.0, 0),
                    (3, 1, 3, 10.0, 4.0, 0.0, 1),
                    (4, 1, 4, 8.0, 5.0, 0.0, 0),
                    (0, 2, 0, 10.0, 1.0, 0.0, 0),
                    (1, 2, 1, 10.0, 2.0, 0.0, 1),
                    (2, 2, 2, 10.0, 3.0, 0.0, 0),
                    (3, 2, 3, 10.0, 4.0, 0.0, 1),
                    (4, 2, 4, 10.0, 5.0, 0.0, 0),
                ],
                dtype=order_dt,
            ),
        )
        assert type(pf._init_cash) == np.ndarray
        base_pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            np.asarray(10.0),
            row_wise=test_row_wise,
            init_cash=np.inf,
            flexible=test_flexible,
        )
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            np.asarray(10.0),
            row_wise=test_row_wise,
            init_cash=InitCashMode.Auto,
            flexible=test_flexible,
        )
        assert_records_close(pf.order_records, base_pf.orders.values)
        assert pf._init_cash == InitCashMode.Auto
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            np.asarray(10.0),
            row_wise=test_row_wise,
            init_cash=InitCashMode.AutoAlign,
            flexible=test_flexible,
        )
        assert_records_close(pf.order_records, base_pf.orders.values)
        assert pf._init_cash == InitCashMode.AutoAlign

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_init_position(self, test_row_wise, test_flexible):

        pos_record_arr = np.empty(1, dtype=trade_dt)

        def pre_segment_func_nb(c):
            pos_record_arr[:] = c.last_pos_record[:]
            return ()

        if test_flexible:

            def order_func_nb(c):
                if c.call_idx < c.group_len:
                    return c.from_col + c.call_idx, nb.order_nb(-np.inf, direction=Direction.LongOnly)
                return -1, nb.order_nothing_nb()

        else:

            def order_func_nb(c):
                return nb.order_nb(-np.inf, direction=Direction.LongOnly)

        pf = vbt.Portfolio.from_order_func(
            1,
            order_func_nb,
            open=0.5,
            jitted=False,
            init_cash=0.0,
            init_position=1.0,
            init_price=0.5,
            pre_segment_func_nb=pre_segment_func_nb,
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        assert pf.init_position == 1.0
        assert_records_close(pf.order_records, np.array([(0, 0, 0, 1.0, 1.0, 0.0, 1)], dtype=order_dt))
        assert_records_close(
            pos_record_arr,
            np.array([(0, 0, 1.0, -1, 0.5, 0.0, -1, np.nan, 0.0, 0.0, 0.0, 0, 0, 0)], dtype=trade_dt),
        )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_cash_earnings(self, test_row_wise, test_flexible):

        if test_flexible:

            @njit
            def order_func_nb(c):
                if c.call_idx < c.group_len:
                    return c.from_col + c.call_idx, nb.order_nb()
                return -1, nb.order_nothing_nb()

        else:

            @njit
            def order_func_nb(c):
                return nb.order_nb()

        pf = vbt.Portfolio.from_order_func(
            1,
            order_func_nb,
            cash_earnings=np.array([0, 1, 2, 3]),
            row_wise=test_row_wise,
            flexible=test_flexible,
        )
        pd.testing.assert_series_equal(pf.cash_earnings, pd.Series([0, 1, 2, 3]))
        assert_records_close(
            pf.order_records,
            np.array(
                [(0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 2, 1.0, 1.0, 0.0, 0), (2, 0, 3, 2.0, 1.0, 0.0, 0)],
                dtype=order_dt,
            ),
        )

    def test_func_calls(self):
        @njit
        def pre_sim_func_nb(c, call_i, pre_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_sim_func_nb(c, call_i, post_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_group_func_nb(c, call_i, pre_group_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_group_func_nb(c, call_i, post_group_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_segment_func_nb(c, call_i, pre_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_segment_func_nb(c, call_i, post_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def order_func_nb(c, call_i, order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            order_lst.append(call_i[0])
            return NoOrder

        @njit
        def post_order_func_nb(c, call_i, post_order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_order_lst.append(call_i[0])

        sub_arg = vbt.RepEval("np.prod([target_shape[0], target_shape[1]])")

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            row_wise=False,
            template_context=dict(np=np),
        )
        assert call_i[0] == 56
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [56]
        assert list(pre_group_lst) == [2, 34]
        assert list(post_group_lst) == [33, 55]
        assert list(pre_segment_lst) == [3, 9, 15, 21, 27, 35, 39, 43, 47, 51]
        assert list(post_segment_lst) == [8, 14, 20, 26, 32, 38, 42, 46, 50, 54]
        assert list(order_lst) == [4, 6, 10, 12, 16, 18, 22, 24, 28, 30, 36, 40, 44, 48, 52]
        assert list(post_order_lst) == [5, 7, 11, 13, 17, 19, 23, 25, 29, 31, 37, 41, 45, 49, 53]

        segment_mask = np.array(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=True,
            call_post_segment=True,
            row_wise=False,
            template_context=dict(np=np),
        )
        assert call_i[0] == 38
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [38]
        assert list(pre_group_lst) == [2, 22]
        assert list(post_group_lst) == [21, 37]
        assert list(pre_segment_lst) == [3, 5, 7, 13, 19, 23, 25, 29, 31, 35]
        assert list(post_segment_lst) == [4, 6, 12, 18, 20, 24, 28, 30, 34, 36]
        assert list(order_lst) == [8, 10, 14, 16, 26, 32]
        assert list(post_order_lst) == [9, 11, 15, 17, 27, 33]

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=False,
            call_post_segment=False,
            row_wise=False,
            template_context=dict(np=np),
        )
        assert call_i[0] == 26
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [26]
        assert list(pre_group_lst) == [2, 16]
        assert list(post_group_lst) == [15, 25]
        assert list(pre_segment_lst) == [3, 9, 17, 21]
        assert list(post_segment_lst) == [8, 14, 20, 24]
        assert list(order_lst) == [4, 6, 10, 12, 18, 22]
        assert list(post_order_lst) == [5, 7, 11, 13, 19, 23]

    def test_func_calls_flexible(self):
        @njit
        def pre_sim_func_nb(c, call_i, pre_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_sim_func_nb(c, call_i, post_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_group_func_nb(c, call_i, pre_group_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_group_func_nb(c, call_i, post_group_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_segment_func_nb(c, call_i, pre_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_segment_func_nb(c, call_i, post_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def flex_order_func_nb(c, call_i, order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            order_lst.append(call_i[0])
            col = c.from_col + c.call_idx
            if c.call_idx < c.group_len:
                return col, NoOrder
            return -1, NoOrder

        @njit
        def post_order_func_nb(c, call_i, post_order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_order_lst.append(call_i[0])

        sub_arg = vbt.RepEval("np.prod([target_shape[0], target_shape[1]])")

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            row_wise=False,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 66
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [66]
        assert list(pre_group_lst) == [2, 39]
        assert list(post_group_lst) == [38, 65]
        assert list(pre_segment_lst) == [3, 10, 17, 24, 31, 40, 45, 50, 55, 60]
        assert list(post_segment_lst) == [9, 16, 23, 30, 37, 44, 49, 54, 59, 64]
        assert list(order_lst) == [
            4,
            6,
            8,
            11,
            13,
            15,
            18,
            20,
            22,
            25,
            27,
            29,
            32,
            34,
            36,
            41,
            43,
            46,
            48,
            51,
            53,
            56,
            58,
            61,
            63,
        ]
        assert list(post_order_lst) == [5, 7, 12, 14, 19, 21, 26, 28, 33, 35, 42, 47, 52, 57, 62]

        segment_mask = np.array(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=True,
            call_post_segment=True,
            row_wise=False,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 42
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [42]
        assert list(pre_group_lst) == [2, 24]
        assert list(post_group_lst) == [23, 41]
        assert list(pre_segment_lst) == [3, 5, 7, 14, 21, 25, 27, 32, 34, 39]
        assert list(post_segment_lst) == [4, 6, 13, 20, 22, 26, 31, 33, 38, 40]
        assert list(order_lst) == [8, 10, 12, 15, 17, 19, 28, 30, 35, 37]
        assert list(post_order_lst) == [9, 11, 16, 18, 29, 36]

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_group_lst = List.empty_list(typeof(0))
        post_group_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_group_func_nb=pre_group_func_nb,
            pre_group_args=(pre_group_lst, sub_arg),
            post_group_func_nb=post_group_func_nb,
            post_group_args=(post_group_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=False,
            call_post_segment=False,
            row_wise=False,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 30
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [30]
        assert list(pre_group_lst) == [2, 18]
        assert list(post_group_lst) == [17, 29]
        assert list(pre_segment_lst) == [3, 10, 19, 24]
        assert list(post_segment_lst) == [9, 16, 23, 28]
        assert list(order_lst) == [4, 6, 8, 11, 13, 15, 20, 22, 25, 27]
        assert list(post_order_lst) == [5, 7, 12, 14, 21, 26]

    def test_func_calls_row_wise(self):
        @njit
        def pre_sim_func_nb(c, call_i, pre_sim_lst):
            call_i[0] += 1
            pre_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_sim_func_nb(c, call_i, post_sim_lst):
            call_i[0] += 1
            post_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_row_func_nb(c, call_i, pre_row_lst):
            call_i[0] += 1
            pre_row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_row_func_nb(c, call_i, post_row_lst):
            call_i[0] += 1
            post_row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_segment_func_nb(c, call_i, pre_segment_lst):
            call_i[0] += 1
            pre_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_segment_func_nb(c, call_i, post_segment_lst):
            call_i[0] += 1
            post_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def order_func_nb(c, call_i, order_lst):
            call_i[0] += 1
            order_lst.append(call_i[0])
            return NoOrder

        @njit
        def post_order_func_nb(c, call_i, post_order_lst):
            call_i[0] += 1
            post_order_lst.append(call_i[0])

        sub_arg = vbt.RepEval("np.prod([target_shape[0], target_shape[1]])")

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst,),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst,),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst,),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst,),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst,),
            row_wise=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 62
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [62]
        assert list(pre_row_lst) == [2, 14, 26, 38, 50]
        assert list(post_row_lst) == [13, 25, 37, 49, 61]
        assert list(pre_segment_lst) == [3, 9, 15, 21, 27, 33, 39, 45, 51, 57]
        assert list(post_segment_lst) == [8, 12, 20, 24, 32, 36, 44, 48, 56, 60]
        assert list(order_lst) == [4, 6, 10, 16, 18, 22, 28, 30, 34, 40, 42, 46, 52, 54, 58]
        assert list(post_order_lst) == [5, 7, 11, 17, 19, 23, 29, 31, 35, 41, 43, 47, 53, 55, 59]

        segment_mask = np.array(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst,),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst,),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst,),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst,),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst,),
            segment_mask=segment_mask,
            call_pre_segment=True,
            call_post_segment=True,
            row_wise=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 44
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [44]
        assert list(pre_row_lst) == [2, 8, 16, 26, 38]
        assert list(post_row_lst) == [7, 15, 25, 37, 43]
        assert list(pre_segment_lst) == [3, 5, 9, 11, 17, 23, 27, 33, 39, 41]
        assert list(post_segment_lst) == [4, 6, 10, 14, 22, 24, 32, 36, 40, 42]
        assert list(order_lst) == [12, 18, 20, 28, 30, 34]
        assert list(post_order_lst) == [13, 19, 21, 29, 31, 35]

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            order_func_nb,
            order_lst,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst,),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst,),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst,),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst,),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst,),
            segment_mask=segment_mask,
            call_pre_segment=False,
            call_post_segment=False,
            row_wise=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 32
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [32]
        assert list(pre_row_lst) == [2, 4, 10, 18, 30]
        assert list(post_row_lst) == [3, 9, 17, 29, 31]
        assert list(pre_segment_lst) == [5, 11, 19, 25]
        assert list(post_segment_lst) == [8, 16, 24, 28]
        assert list(order_lst) == [6, 12, 14, 20, 22, 26]
        assert list(post_order_lst) == [7, 13, 15, 21, 23, 27]

    def test_func_calls_row_wise_flexible(self):
        @njit
        def pre_sim_func_nb(c, call_i, pre_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_sim_func_nb(c, call_i, post_sim_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_row_func_nb(c, call_i, pre_row_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_row_func_nb(c, call_i, post_row_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def pre_segment_func_nb(c, call_i, pre_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            pre_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def post_segment_func_nb(c, call_i, post_segment_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def flex_order_func_nb(c, call_i, order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            order_lst.append(call_i[0])
            col = c.from_col + c.call_idx
            if c.call_idx < c.group_len:
                return col, NoOrder
            return -1, NoOrder

        @njit
        def post_order_func_nb(c, call_i, post_order_lst, sub_arg):
            if sub_arg != 15:
                raise ValueError
            call_i[0] += 1
            post_order_lst.append(call_i[0])

        sub_arg = vbt.RepEval("np.prod([target_shape[0], target_shape[1]])")

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst, sub_arg),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            row_wise=True,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 72
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [72]
        assert list(pre_row_lst) == [2, 16, 30, 44, 58]
        assert list(post_row_lst) == [15, 29, 43, 57, 71]
        assert list(pre_segment_lst) == [3, 10, 17, 24, 31, 38, 45, 52, 59, 66]
        assert list(post_segment_lst) == [9, 14, 23, 28, 37, 42, 51, 56, 65, 70]
        assert list(order_lst) == [
            4,
            6,
            8,
            11,
            13,
            18,
            20,
            22,
            25,
            27,
            32,
            34,
            36,
            39,
            41,
            46,
            48,
            50,
            53,
            55,
            60,
            62,
            64,
            67,
            69,
        ]
        assert list(post_order_lst) == [5, 7, 12, 19, 21, 26, 33, 35, 40, 47, 49, 54, 61, 63, 68]

        segment_mask = np.array(
            [
                [False, False],
                [False, True],
                [True, False],
                [True, True],
                [False, False],
            ]
        )
        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst, sub_arg),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=True,
            call_post_segment=True,
            row_wise=True,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 48
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [48]
        assert list(pre_row_lst) == [2, 8, 17, 28, 42]
        assert list(post_row_lst) == [7, 16, 27, 41, 47]
        assert list(pre_segment_lst) == [3, 5, 9, 11, 18, 25, 29, 36, 43, 45]
        assert list(post_segment_lst) == [4, 6, 10, 15, 24, 26, 35, 40, 44, 46]
        assert list(order_lst) == [12, 14, 19, 21, 23, 30, 32, 34, 37, 39]
        assert list(post_order_lst) == [13, 20, 22, 31, 33, 38]

        call_i = np.array([0])
        pre_sim_lst = List.empty_list(typeof(0))
        post_sim_lst = List.empty_list(typeof(0))
        pre_row_lst = List.empty_list(typeof(0))
        post_row_lst = List.empty_list(typeof(0))
        pre_segment_lst = List.empty_list(typeof(0))
        post_segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        post_order_lst = List.empty_list(typeof(0))
        vbt.Portfolio.from_order_func(
            price_wide,
            flex_order_func_nb,
            order_lst,
            sub_arg,
            group_by=np.array([0, 0, 1]),
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(call_i, pre_sim_lst, sub_arg),
            post_sim_func_nb=post_sim_func_nb,
            post_sim_args=(call_i, post_sim_lst, sub_arg),
            pre_row_func_nb=pre_row_func_nb,
            pre_row_args=(pre_row_lst, sub_arg),
            post_row_func_nb=post_row_func_nb,
            post_row_args=(post_row_lst, sub_arg),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(pre_segment_lst, sub_arg),
            post_segment_func_nb=post_segment_func_nb,
            post_segment_args=(post_segment_lst, sub_arg),
            post_order_func_nb=post_order_func_nb,
            post_order_args=(post_order_lst, sub_arg),
            segment_mask=segment_mask,
            call_pre_segment=False,
            call_post_segment=False,
            row_wise=True,
            flexible=True,
            template_context=dict(np=np),
        )
        assert call_i[0] == 36
        assert list(pre_sim_lst) == [1]
        assert list(post_sim_lst) == [36]
        assert list(pre_row_lst) == [2, 4, 11, 20, 34]
        assert list(post_row_lst) == [3, 10, 19, 33, 35]
        assert list(pre_segment_lst) == [5, 12, 21, 28]
        assert list(post_segment_lst) == [9, 18, 27, 32]
        assert list(order_lst) == [6, 8, 13, 15, 17, 22, 24, 26, 29, 31]
        assert list(post_order_lst) == [7, 14, 16, 23, 25, 30]

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_max_orders(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        assert (
            vbt.Portfolio.from_order_func(
                price_wide,
                order_func,
                np.asarray(np.inf),
                row_wise=test_row_wise,
                flexible=test_flexible,
            ).order_records.shape[0]
            == 15
        )
        assert (
            vbt.Portfolio.from_order_func(
                price_wide,
                order_func,
                np.asarray(np.inf),
                row_wise=test_row_wise,
                max_orders=5,
                flexible=test_flexible,
            ).order_records.shape[0]
            == 15
        )
        assert (
            vbt.Portfolio.from_order_func(
                price_wide,
                order_func,
                np.asarray(np.inf),
                row_wise=test_row_wise,
                max_orders=0,
                flexible=test_flexible,
            ).order_records.shape[0]
            == 0
        )
        with pytest.raises(Exception):
            vbt.Portfolio.from_order_func(
                price_wide,
                order_func,
                np.asarray(np.inf),
                row_wise=test_row_wise,
                max_orders=4,
                flexible=test_flexible,
            )

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_max_logs(self, test_row_wise, test_flexible):
        log_order_func = log_flex_order_func_nb if test_flexible else log_order_func_nb
        assert (
            vbt.Portfolio.from_order_func(
                price_wide,
                log_order_func,
                np.asarray(np.inf),
                row_wise=test_row_wise,
                flexible=test_flexible,
            ).log_records.shape[0]
            == 15
        )
        assert (
            vbt.Portfolio.from_order_func(
                price_wide,
                log_order_func,
                np.asarray(np.inf),
                row_wise=test_row_wise,
                max_logs=5,
                flexible=test_flexible,
            ).log_records.shape[0]
            == 15
        )
        assert (
            vbt.Portfolio.from_order_func(
                price_wide,
                log_order_func,
                np.asarray(np.inf),
                row_wise=test_row_wise,
                max_logs=0,
                flexible=test_flexible,
            ).log_records.shape[0]
            == 0
        )
        with pytest.raises(Exception):
            vbt.Portfolio.from_order_func(
                price_wide,
                log_order_func,
                np.asarray(np.inf),
                row_wise=test_row_wise,
                max_logs=4,
                flexible=test_flexible,
            )

    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_jitted_parallel(self, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        pf = vbt.Portfolio.from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            group_by=group_by,
            row_wise=False,
            flexible=test_flexible,
            jitted=dict(parallel=True),
        )
        pf2 = vbt.Portfolio.from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            group_by=group_by,
            row_wise=False,
            flexible=test_flexible,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        pf = vbt.Portfolio.from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            group_by=group_by,
            cash_sharing=True,
            row_wise=False,
            flexible=test_flexible,
            jitted=dict(parallel=True),
        )
        pf2 = vbt.Portfolio.from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            group_by=group_by,
            cash_sharing=True,
            row_wise=False,
            flexible=test_flexible,
            jitted=dict(parallel=False),
        )
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_chunked(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb
        price_wide2 = price_wide.copy()
        price_wide2.iloc[:, 1] *= 0.9
        price_wide2.iloc[:, 2] *= 1.1
        chunked = dict(
            arg_take_spec=dict(
                order_args=vbt.ArgsTaker(
                    vbt.FlexArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query="group_lens")),
                ),
                flex_order_args=vbt.ArgsTaker(
                    vbt.FlexArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query="group_lens")),
                ),
            )
        )
        pf = vbt.Portfolio.from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            group_by=group_by,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=chunked,
        )
        pf2 = vbt.Portfolio.from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            group_by=group_by,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=False,
        )
        if test_row_wise:
            pd.testing.assert_series_equal(pf.total_profit, pf2.total_profit)
            pd.testing.assert_series_equal(pf.total_profit, pf2.total_profit)
        else:
            assert_records_close(pf.order_records, pf2.order_records)
            assert_records_close(pf.log_records, pf2.log_records)
        pf = vbt.Portfolio.from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            group_by=group_by,
            cash_sharing=True,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=chunked,
        )
        pf2 = vbt.Portfolio.from_order_func(
            price_wide2,
            order_func,
            vbt.Rep("size"),
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            group_by=group_by,
            cash_sharing=True,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=False,
        )
        if test_row_wise:
            pd.testing.assert_series_equal(pf.total_profit, pf2.total_profit)
            pd.testing.assert_series_equal(pf.total_profit, pf2.total_profit)
        else:
            assert_records_close(pf.order_records, pf2.order_records)
            assert_records_close(pf.log_records, pf2.log_records)

    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    def test_in_outputs(self, test_row_wise, test_flexible):
        order_func = flex_order_func_nb if test_flexible else order_func_nb

        @njit
        def post_sim_func_nb(c):
            c.in_outputs.custom_1d_arr[:] = 10.0
            c.in_outputs.custom_2d_arr[:] = 100
            c.in_outputs.custom_rec_arr["col"][:5] = 0
            c.in_outputs.custom_rec_arr["col"][5:10] = 1
            c.in_outputs.custom_rec_arr["col"][10:15] = 2

        class CustomMapper(vbt.ChunkMapper):
            def map(self, chunk_meta, ann_args=None, **kwargs):
                mapper = vbt.GroupLensMapper(arg_query="group_lens")
                chunk_meta = mapper.apply(chunk_meta, ann_args=ann_args, **kwargs)
                target_shape = ann_args["target_shape"]["value"]
                new_chunk_meta = vbt.ChunkMeta(
                    uuid=str(uuid.uuid4()),
                    idx=chunk_meta.idx,
                    start=chunk_meta.start * target_shape[0],
                    end=chunk_meta.end * target_shape[0],
                    indices=None,
                )
                return new_chunk_meta

        custom_dtype = np.dtype([("col", np.int_)])
        chunked = dict(
            arg_take_spec=dict(
                order_args=vbt.ArgsTaker(
                    vbt.FlexArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query="group_lens")),
                ),
                flex_order_args=vbt.ArgsTaker(
                    vbt.FlexArraySlicer(axis=1, mapper=vbt.GroupLensMapper(arg_query="group_lens")),
                ),
                in_outputs=vbt.ArgsTaker(
                    vbt.ArraySlicer(axis=0, mapper=vbt.GroupLensMapper(arg_query="group_lens")),
                    vbt.ArraySlicer(axis=1),
                    vbt.ArraySlicer(axis=0, mapper=CustomMapper()),
                ),
            )
        )
        in_outputs = dict(
            custom_1d_arr=vbt.RepEval("np.full(target_shape[1], 0., dtype=np.float_)"),
            custom_2d_arr=vbt.RepEval("np.empty((target_shape[0], len(group_lens)), dtype=np.int_)"),
            custom_rec_arr=vbt.RepEval("np.empty(target_shape[0] * target_shape[1], dtype=custom_dtype)"),
        )
        in_outputs = InOutputs(**in_outputs)
        pf = vbt.Portfolio.from_order_func(
            price_wide,
            order_func,
            vbt.Rep("size"),
            post_sim_func_nb=post_sim_func_nb,
            broadcast_named_args=dict(size=[0, 1, np.inf]),
            in_outputs=in_outputs,
            template_context=dict(custom_dtype=custom_dtype),
            group_by=group_by,
            cash_sharing=False,
            row_wise=test_row_wise,
            flexible=test_flexible,
            chunked=chunked,
        )

        custom_1d_arr = np.array([10.0, 10.0, 10.0])
        custom_2d_arr = np.array([[100, 100], [100, 100], [100, 100], [100, 100], [100, 100]])
        custom_rec_arr = np.array(
            [(0,), (0,), (0,), (0,), (0,), (1,), (1,), (1,), (1,), (1,), (2,), (2,), (2,), (2,), (2,)],
            dtype=custom_dtype,
        )

        np.testing.assert_array_equal(pf.in_outputs.custom_1d_arr, custom_1d_arr)
        np.testing.assert_array_equal(pf.in_outputs.custom_2d_arr, custom_2d_arr)
        np.testing.assert_array_equal(pf.in_outputs.custom_rec_arr, custom_rec_arr)


# ############# from_def_order_func ############# #


class TestFromDefOrderFunc:
    @pytest.mark.parametrize("test_row_wise", [False, True])
    @pytest.mark.parametrize("test_flexible", [False, True])
    @pytest.mark.parametrize("test_jitted", [False, True])
    @pytest.mark.parametrize("test_chunked", [False, True])
    def test_compare(self, test_row_wise, test_flexible, test_jitted, test_chunked):
        pf = vbt.Portfolio.from_def_order_func(
            price_wide,
            size=[0, 1, np.inf],
            log=True,
            row_wise=test_row_wise,
            flexible=test_flexible,
            jitted=test_jitted,
            chunked=test_chunked,
        )
        pf2 = vbt.Portfolio.from_orders(price_wide, size=[0, 1, np.inf], log=True)
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        assert pf.wrapper == pf2.wrapper

        pf = vbt.Portfolio.from_def_order_func(
            price_wide,
            size=[0, 1, np.inf],
            log=True,
            group_by=True,
            cash_sharing=True,
            row_wise=test_row_wise,
            flexible=test_flexible,
            jitted=test_jitted,
            chunked=test_chunked,
        )
        pf2 = vbt.Portfolio.from_orders(price_wide, size=[0, 1, np.inf], log=True, group_by=True, cash_sharing=True)
        assert_records_close(pf.order_records, pf2.order_records)
        assert_records_close(pf.log_records, pf2.log_records)
        assert pf.wrapper == pf2.wrapper

        target_hold_value = pd.DataFrame(
            {"a": [0.0, 70.0, 30.0, 0.0, 70.0], "b": [30.0, 0.0, 70.0, 30.0, 30.0], "c": [70.0, 30.0, 0.0, 70.0, 0.0]},
            index=price.index,
        )
        pd.testing.assert_frame_equal(
            vbt.Portfolio.from_def_order_func(
                close=1.0,
                size=target_hold_value,
                size_type="targetvalue",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                call_seq="auto",
            ).get_asset_value(group_by=False),
            target_hold_value,
        )
        pd.testing.assert_frame_equal(
            vbt.Portfolio.from_def_order_func(
                close=1.0,
                size=target_hold_value / 100,
                size_type="targetpercent",
                group_by=np.array([0, 0, 0]),
                cash_sharing=True,
                call_seq="auto",
            ).get_asset_value(group_by=False),
            target_hold_value,
        )


# ############# Portfolio ############# #

price_na = pd.DataFrame(
    {"a": [np.nan, 2.0, 3.0, 4.0, 5.0], "b": [1.0, 2.0, np.nan, 4.0, 5.0], "c": [1.0, 2.0, 3.0, 4.0, np.nan]},
    index=price.index,
)
order_size_new = pd.DataFrame(
    {"a": [0.0, 0.1, -1.0, -0.1, 1.0], "b": [0.0, 0.1, -1.0, -0.1, 1.0], "c": [1.0, 0.1, -1.0, -0.1, 1.0]},
    index=price.index,
)
bm_price_na = pd.DataFrame(
    {"a": [5.0, 4.0, 3.0, 2.0, np.nan], "b": [5.0, 4.0, np.nan, 2.0, 1.0], "c": [np.nan, 4.0, 3.0, 2.0, 1.0]},
    index=price.index,
)
init_position = [1.0, -1.0, 0.0]
directions = ["longonly", "shortonly", "both"]
group_by = pd.Index(["first", "first", "second"], name="group")

pf_kwargs = dict(
    close=price_na,
    size=order_size_new,
    size_type="amount",
    direction=directions,
    fees=0.01,
    fixed_fees=0.1,
    slippage=0.01,
    log=True,
    call_seq="reversed",
    group_by=None,
    init_cash=[100.0, 100.0, 100.0],
    init_position=init_position,
    init_price=price_na.bfill().values[0],
    cash_deposits=pd.DataFrame(
        {"a": [0.0, 0.0, 100.0, 0.0, 0.0], "b": [0.0, 0.0, 100.0, 0.0, 0.0], "c": [0.0, 0.0, 0.0, 0.0, 0.0]},
        index=price.index,
    ),
    freq="1D",
    attach_call_seq=True,
    bm_close=bm_price_na,
)

pf = vbt.Portfolio.from_orders(**pf_kwargs)  # independent
pf_filled = vbt.Portfolio.from_orders(**pf_kwargs, fill_returns=True)

pf_grouped_kwargs = dict(
    close=price_na,
    size=order_size_new,
    size_type="amount",
    direction=directions,
    fees=0.01,
    fixed_fees=0.1,
    slippage=0.01,
    log=True,
    call_seq="reversed",
    group_by=group_by,
    cash_sharing=False,
    init_cash=[100.0, 100.0, 100.0],
    init_position=init_position,
    init_price=price_na.bfill().values[0],
    cash_deposits=pd.DataFrame(
        {"a": [0.0, 0.0, 100.0, 0.0, 0.0], "b": [0.0, 0.0, 100.0, 0.0, 0.0], "c": [0.0, 0.0, 0.0, 0.0, 0.0]},
        index=price.index,
    ),
    freq="1D",
    attach_call_seq=True,
    bm_close=bm_price_na,
)

pf_grouped = vbt.Portfolio.from_orders(**pf_grouped_kwargs)  # grouped
pf_grouped_filled = vbt.Portfolio.from_orders(**pf_grouped_kwargs, fill_returns=True)

pf_shared_kwargs = dict(
    close=price_na,
    size=order_size_new,
    size_type="amount",
    direction=directions,
    fees=0.01,
    fixed_fees=0.1,
    slippage=0.01,
    log=True,
    call_seq="reversed",
    group_by=group_by,
    cash_sharing=True,
    init_cash=[200.0, 100.0],
    init_position=init_position,
    init_price=price_na.bfill().values[0],
    cash_deposits=pd.DataFrame(
        {"first": [0.0, 0.0, 200.0, 0.0, 0.0], "second": [0.0, 0.0, 0.0, 0.0, 0.0]},
        index=price.index,
    ),
    freq="1D",
    attach_call_seq=True,
    bm_close=bm_price_na,
)

pf_shared = vbt.Portfolio.from_orders(**pf_shared_kwargs)  # shared
pf_shared_filled = vbt.Portfolio.from_orders(**pf_shared_kwargs, fill_returns=True)


class TestPortfolio:
    def test_config(self, tmp_path):
        pf2 = pf.copy()
        pf2._metrics = pf2._metrics.copy()
        pf2.metrics["hello"] = "world"
        pf2._subplots = pf2.subplots.copy()
        pf2.subplots["hello"] = "world"
        assert vbt.Portfolio.loads(pf2["a"].dumps()) == pf2["a"]
        assert vbt.Portfolio.loads(pf2.dumps()) == pf2
        pf2.save(tmp_path / "pf")
        assert vbt.Portfolio.load(tmp_path / "pf") == pf2

    def test_wrapper(self):
        pd.testing.assert_index_equal(pf.wrapper.index, price_na.index)
        pd.testing.assert_index_equal(pf.wrapper.columns, price_na.columns)
        assert pf.wrapper.ndim == 2
        assert pf.wrapper.grouper.group_by is None
        assert pf.wrapper.grouper.allow_enable
        assert pf.wrapper.grouper.allow_disable
        assert pf.wrapper.grouper.allow_modify
        pd.testing.assert_index_equal(pf_grouped.wrapper.index, price_na.index)
        pd.testing.assert_index_equal(pf_grouped.wrapper.columns, price_na.columns)
        assert pf_grouped.wrapper.ndim == 2
        pd.testing.assert_index_equal(pf_grouped.wrapper.grouper.group_by, group_by)
        assert pf_grouped.wrapper.grouper.allow_enable
        assert pf_grouped.wrapper.grouper.allow_disable
        assert pf_grouped.wrapper.grouper.allow_modify
        pd.testing.assert_index_equal(pf_shared.wrapper.index, price_na.index)
        pd.testing.assert_index_equal(pf_shared.wrapper.columns, price_na.columns)
        assert pf_shared.wrapper.ndim == 2
        pd.testing.assert_index_equal(pf_shared.wrapper.grouper.group_by, group_by)
        assert not pf_shared.wrapper.grouper.allow_enable
        assert pf_shared.wrapper.grouper.allow_disable
        assert not pf_shared.wrapper.grouper.allow_modify

    def test_indexing(self):
        assert pf["a"].wrapper == pf.wrapper["a"]
        assert pf["a"].orders == pf.orders["a"]
        assert pf["a"].logs == pf.logs["a"]
        assert pf["a"].init_cash == pf.init_cash["a"]
        pd.testing.assert_series_equal(pf["a"].call_seq, pf.call_seq["a"])
        pd.testing.assert_series_equal(pf["a"].cash_deposits, pf.cash_deposits["a"])

        assert pf["c"].wrapper == pf.wrapper["c"]
        assert pf["c"].orders == pf.orders["c"]
        assert pf["c"].logs == pf.logs["c"]
        assert pf["c"].init_cash == pf.init_cash["c"]
        pd.testing.assert_series_equal(pf["c"].call_seq, pf.call_seq["c"])
        pd.testing.assert_series_equal(pf["c"].cash_deposits, pf.cash_deposits["c"])

        assert pf[["c"]].wrapper == pf.wrapper[["c"]]
        assert pf[["c"]].orders == pf.orders[["c"]]
        assert pf[["c"]].logs == pf.logs[["c"]]
        pd.testing.assert_series_equal(pf[["c"]].init_cash, pf.init_cash[["c"]])
        pd.testing.assert_frame_equal(pf[["c"]].call_seq, pf.call_seq[["c"]])
        pd.testing.assert_frame_equal(pf[["c"]].cash_deposits, pf.cash_deposits[["c"]])

        assert pf_grouped["first"].wrapper == pf_grouped.wrapper["first"]
        assert pf_grouped["first"].orders == pf_grouped.orders["first"]
        assert pf_grouped["first"].logs == pf_grouped.logs["first"]
        assert pf_grouped["first"].init_cash == pf_grouped.init_cash["first"]
        pd.testing.assert_frame_equal(pf_grouped["first"].call_seq, pf_grouped.call_seq[["a", "b"]])
        pd.testing.assert_series_equal(pf_grouped["first"].cash_deposits, pf_grouped.cash_deposits["first"])

        assert pf_grouped[["first"]].wrapper == pf_grouped.wrapper[["first"]]
        assert pf_grouped[["first"]].orders == pf_grouped.orders[["first"]]
        assert pf_grouped[["first"]].logs == pf_grouped.logs[["first"]]
        pd.testing.assert_series_equal(pf_grouped[["first"]].init_cash, pf_grouped.init_cash[["first"]])
        pd.testing.assert_frame_equal(pf_grouped[["first"]].call_seq, pf_grouped.call_seq[["a", "b"]])
        pd.testing.assert_frame_equal(pf_grouped[["first"]].cash_deposits, pf_grouped.cash_deposits[["first"]])

        assert pf_grouped["second"].wrapper == pf_grouped.wrapper["second"]
        assert pf_grouped["second"].orders == pf_grouped.orders["second"]
        assert pf_grouped["second"].logs == pf_grouped.logs["second"]
        assert pf_grouped["second"].init_cash == pf_grouped.init_cash["second"]
        pd.testing.assert_series_equal(pf_grouped["second"].call_seq, pf_grouped.call_seq["c"])
        pd.testing.assert_series_equal(pf_grouped["second"].cash_deposits, pf_grouped.cash_deposits["second"])

        assert pf_grouped[["second"]].orders == pf_grouped.orders[["second"]]
        assert pf_grouped[["second"]].wrapper == pf_grouped.wrapper[["second"]]
        assert pf_grouped[["second"]].orders == pf_grouped.orders[["second"]]
        assert pf_grouped[["second"]].logs == pf_grouped.logs[["second"]]
        pd.testing.assert_series_equal(pf_grouped[["second"]].init_cash, pf_grouped.init_cash[["second"]])
        pd.testing.assert_frame_equal(pf_grouped[["second"]].call_seq, pf_grouped.call_seq[["c"]])
        pd.testing.assert_frame_equal(pf_grouped[["second"]].cash_deposits, pf_grouped.cash_deposits[["second"]])

        assert pf_shared["first"].wrapper == pf_shared.wrapper["first"]
        assert pf_shared["first"].orders == pf_shared.orders["first"]
        assert pf_shared["first"].logs == pf_shared.logs["first"]
        assert pf_shared["first"].init_cash == pf_shared.init_cash["first"]
        pd.testing.assert_frame_equal(pf_shared["first"].call_seq, pf_shared.call_seq[["a", "b"]])
        pd.testing.assert_series_equal(pf_shared["first"].cash_deposits, pf_shared.cash_deposits["first"])

        assert pf_shared[["first"]].orders == pf_shared.orders[["first"]]
        assert pf_shared[["first"]].wrapper == pf_shared.wrapper[["first"]]
        assert pf_shared[["first"]].orders == pf_shared.orders[["first"]]
        assert pf_shared[["first"]].logs == pf_shared.logs[["first"]]
        pd.testing.assert_series_equal(pf_shared[["first"]].init_cash, pf_shared.init_cash[["first"]])
        pd.testing.assert_frame_equal(pf_shared[["first"]].call_seq, pf_shared.call_seq[["a", "b"]])
        pd.testing.assert_frame_equal(pf_shared[["first"]].cash_deposits, pf_shared.cash_deposits[["first"]])

        assert pf_shared["second"].wrapper == pf_shared.wrapper["second"]
        assert pf_shared["second"].orders == pf_shared.orders["second"]
        assert pf_shared["second"].logs == pf_shared.logs["second"]
        assert pf_shared["second"].init_cash == pf_shared.init_cash["second"]
        pd.testing.assert_series_equal(pf_shared["second"].call_seq, pf_shared.call_seq["c"])
        pd.testing.assert_series_equal(pf_shared["second"].cash_deposits, pf_shared.cash_deposits["second"])

        assert pf_shared[["second"]].wrapper == pf_shared.wrapper[["second"]]
        assert pf_shared[["second"]].orders == pf_shared.orders[["second"]]
        assert pf_shared[["second"]].logs == pf_shared.logs[["second"]]
        pd.testing.assert_series_equal(pf_shared[["second"]].init_cash, pf_shared.init_cash[["second"]])
        pd.testing.assert_frame_equal(pf_shared[["second"]].call_seq, pf_shared.call_seq[["c"]])
        pd.testing.assert_frame_equal(pf_shared[["second"]].cash_deposits, pf_shared.cash_deposits[["second"]])

    def test_regroup(self):
        assert pf.regroup(None) == pf
        assert pf.regroup(False) == pf
        assert pf.regroup(group_by) != pf
        pd.testing.assert_index_equal(pf.regroup(group_by).wrapper.grouper.group_by, group_by)
        assert pf_grouped.regroup(None) == pf_grouped
        assert pf_grouped.regroup(False) != pf_grouped
        assert pf_grouped.regroup(False).wrapper.grouper.group_by is None
        assert pf_grouped.regroup(group_by) == pf_grouped
        assert pf_shared.regroup(None) == pf_shared
        with pytest.raises(Exception):
            pf_shared.regroup(False)
        assert pf_shared.regroup(group_by) == pf_shared

    def test_cash_sharing(self):
        assert not pf.cash_sharing
        assert not pf_grouped.cash_sharing
        assert pf_shared.cash_sharing

    def test_call_seq(self):
        pd.testing.assert_frame_equal(
            pf.call_seq,
            pd.DataFrame(
                np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.call_seq,
            pd.DataFrame(
                np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.call_seq,
            pd.DataFrame(
                np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )

    def test_in_outputs(self):
        in_outputs = dict(
            init_cash=np.arange(3),
            init_position_value=np.arange(3),
            close=np.arange(15).reshape((5, 3)),
            cash_flow=np.arange(15).reshape((5, 3)),
            orders=np.concatenate(
                (np.full(5, 0, dtype=order_dt), np.full(5, 1, dtype=order_dt), np.full(5, 2, dtype=order_dt)),
            ),
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf2 = pf.replace(in_outputs=in_outputs)

        np.testing.assert_array_equal(pf2.init_cash.values, in_outputs.init_cash)
        np.testing.assert_array_equal(pf2.init_position_value.values, in_outputs.init_position_value)
        np.testing.assert_array_equal(pf2.close.values, in_outputs.close)
        np.testing.assert_array_equal(pf2.cash_flow.values, in_outputs.cash_flow)
        np.testing.assert_array_equal(pf2.orders.values, in_outputs.orders)

        assert pf2["a"].init_cash == in_outputs.init_cash[0]
        assert pf2["a"].init_position_value == in_outputs.init_position_value[0]
        np.testing.assert_array_equal(pf2["a"].close.values, in_outputs.close[:, 0])
        np.testing.assert_array_equal(pf2["a"].cash_flow.values, in_outputs.cash_flow[:, 0])
        np.testing.assert_array_equal(pf2["a"].orders.values, in_outputs.orders[:5])

        in_outputs = dict(
            init_cash=np.arange(2),
            init_position_value=np.arange(3),
            close=np.arange(15).reshape((5, 3)),
            cash_flow=np.arange(10).reshape((5, 2)),
            orders=np.concatenate(
                (np.full(5, 0, dtype=order_dt), np.full(5, 1, dtype=order_dt), np.full(5, 2, dtype=order_dt)),
            ),
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf_shared2 = pf_shared.replace(in_outputs=in_outputs)

        np.testing.assert_array_equal(pf_shared2.init_cash.values, in_outputs.init_cash)
        np.testing.assert_array_equal(pf_shared2.init_position_value.values, in_outputs.init_position_value)
        np.testing.assert_array_equal(pf_shared2.close.values, in_outputs.close)
        np.testing.assert_array_equal(pf_shared2.cash_flow.values, in_outputs.cash_flow)
        np.testing.assert_array_equal(pf_shared2.orders.values, in_outputs.orders)

        assert pf_shared2["first"].init_cash == in_outputs.init_cash[0]
        np.testing.assert_array_equal(
            pf_shared2["first"].init_position_value.values,
            in_outputs.init_position_value[:2],
        )
        np.testing.assert_array_equal(pf_shared2["first"].close.values, in_outputs.close[:, :2])
        np.testing.assert_array_equal(pf_shared2["first"].cash_flow.values, in_outputs.cash_flow[:, 0])
        np.testing.assert_array_equal(pf_shared2["first"].orders.values, in_outputs.orders[:10])

        def create_in_outputs(**kwargs):
            return namedtuple("InOutputs", kwargs)(**kwargs)

        np.testing.assert_array_equal(
            pf_shared.replace(in_outputs=create_in_outputs(init_cash_pg=np.arange(2))).init_cash.values,
            np.arange(2),
        )
        np.testing.assert_array_equal(
            pf_shared.replace(in_outputs=create_in_outputs(init_cash_pcg=np.arange(2))).init_cash.values,
            np.arange(2),
        )
        np.testing.assert_array_equal(
            pf_shared.replace(in_outputs=create_in_outputs(init_cash_cs=np.arange(2))).init_cash.values,
            np.arange(2),
        )
        np.testing.assert_array_equal(
            pf_shared.replace(
                in_outputs=create_in_outputs(init_position_value_pc=np.arange(3))
            ).init_position_value.values,
            np.arange(3),
        )
        np.testing.assert_array_equal(
            pf_grouped.replace(
                in_outputs=create_in_outputs(init_position_value_cs=np.arange(3))
            ).init_position_value.values,
            np.arange(3),
        )
        np.testing.assert_array_equal(
            pf.replace(in_outputs=create_in_outputs(init_cash_pc=np.arange(3))).init_cash.values,
            np.arange(3),
        )
        np.testing.assert_array_equal(
            pf.replace(in_outputs=create_in_outputs(init_cash_pcg=np.arange(3))).init_cash.values,
            np.arange(3),
        )
        np.testing.assert_array_equal(
            pf.replace(in_outputs=create_in_outputs(init_cash_cs=np.arange(3))).init_cash.values,
            np.arange(3),
        )

    def test_custom_in_outputs(self):
        in_outputs = dict(
            arr_1d_cs=np.arange(3),
            arr_2d_cs=np.arange(15).reshape((5, 3)),
            arr_1d_pcg=np.arange(3),
            arr_2d_pcg=np.arange(15).reshape((5, 3)),
            arr_1d_pg=np.arange(3),
            arr_2d_pg=np.arange(15).reshape((5, 3)),
            arr_1d_pc=np.arange(3),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d=np.arange(3),
            arr_2d=np.arange(15).reshape((5, 3)),
            arr_records=np.concatenate(
                (
                    np.full(5, 0, dtype=np.dtype([("col", np.int_)])),
                    np.full(5, 1, dtype=np.dtype([("col", np.int_)])),
                    np.full(5, 2, dtype=np.dtype([("col", np.int_)])),
                )
            ),
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf2 = pf.replace(in_outputs=in_outputs)

        with pytest.raises(AttributeError):
            pf2.get_in_output("my_arr")

        pd.testing.assert_series_equal(
            pf2.get_in_output("arr_1d_cs"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_cs, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(pf2.get_in_output("arr_2d_cs"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_cs))
        pd.testing.assert_series_equal(
            pf2.get_in_output("arr_1d_pcg"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(pf2.get_in_output("arr_2d_pcg"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_pcg))
        pd.testing.assert_series_equal(
            pf2.get_in_output("arr_1d_pg"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_pg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(pf2.get_in_output("arr_2d_pg"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_pg))
        pd.testing.assert_series_equal(
            pf2.get_in_output("arr_1d_pc"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_pc, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(pf2.get_in_output("arr_2d_pc"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_pc))
        pd.testing.assert_series_equal(
            pf2.get_in_output("arr_1d"),
            pf2.wrapper.wrap_reduced(pf2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(pf2.get_in_output("arr_2d"), pf2.wrapper.wrap(pf2.in_outputs.arr_2d_pcg))
        assert_records_close(pf2.get_in_output("arr_records"), pf2.in_outputs.arr_records)
        with pytest.raises(NotImplementedError):
            pf2.get_in_output("arr_records", force_wrapping=True)

        assert pf2["a"].in_outputs.arr_1d_cs == in_outputs.arr_1d_cs[0]
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_2d_cs, in_outputs.arr_2d_cs[:, 0])
        assert pf2["a"].in_outputs.arr_1d_pcg == in_outputs.arr_1d_pcg[0]
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_2d_pcg, in_outputs.arr_2d_pcg[:, 0])
        assert pf2["a"].in_outputs.arr_1d_pg == in_outputs.arr_1d_pg[0]
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_2d_pg, in_outputs.arr_2d_pg[:, 0])
        assert pf2["a"].in_outputs.arr_1d_pc == in_outputs.arr_1d_pc[0]
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_2d_pc, in_outputs.arr_2d_pc[:, 0])
        np.testing.assert_array_equal(pf2["a"].in_outputs.arr_records, in_outputs.arr_records[:5])

        in_outputs = dict(
            arr_1d_cs=np.arange(3),
            arr_2d_cs=np.arange(15).reshape((5, 3)),
            arr_1d_pcg=np.arange(2),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_1d_pg=np.arange(2),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_1d_pc=np.arange(3),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d=np.arange(2),
            arr_2d=np.arange(10).reshape((5, 2)),
            arr_records=np.concatenate(
                (
                    np.full(5, 0, dtype=np.dtype([("col", np.int_)])),
                    np.full(5, 1, dtype=np.dtype([("col", np.int_)])),
                    np.full(5, 2, dtype=np.dtype([("col", np.int_)])),
                )
            ),
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf_grouped2 = pf_grouped.replace(in_outputs=in_outputs)

        pd.testing.assert_series_equal(
            pf_grouped2.get_in_output("arr_1d_cs"),
            pf_grouped2.wrapper.wrap_reduced(pf_grouped2.in_outputs.arr_1d_cs, name_or_index="arr", group_by=False),
        )
        pd.testing.assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d_cs"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_cs, group_by=False),
        )
        pd.testing.assert_series_equal(
            pf_grouped2.get_in_output("arr_1d_pcg"),
            pf_grouped2.wrapper.wrap_reduced(pf_grouped2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d_pcg"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_pcg),
        )
        pd.testing.assert_series_equal(
            pf_grouped2.get_in_output("arr_1d_pg"),
            pf_grouped2.wrapper.wrap_reduced(pf_grouped2.in_outputs.arr_1d_pg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d_pg"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_pg),
        )
        pd.testing.assert_series_equal(
            pf_grouped2.get_in_output("arr_1d_pc"),
            pf_grouped2.wrapper.wrap_reduced(pf_grouped2.in_outputs.arr_1d_pc, name_or_index="arr", group_by=False),
        )
        pd.testing.assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d_pc"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_pc, group_by=False),
        )
        pd.testing.assert_series_equal(
            pf_grouped2.get_in_output("arr_1d"),
            pf_grouped2.wrapper.wrap_reduced(pf_grouped2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(
            pf_grouped2.get_in_output("arr_2d"),
            pf_grouped2.wrapper.wrap(pf_grouped2.in_outputs.arr_2d_pcg),
        )
        assert_records_close(pf_grouped2.get_in_output("arr_records"), pf_grouped2.in_outputs.arr_records)
        with pytest.raises(NotImplementedError):
            pf_grouped2.get_in_output("arr_records", force_wrapping=True)

        np.testing.assert_array_equal(pf_grouped2["first"].in_outputs.arr_1d_cs, in_outputs.arr_1d_cs[:2])
        np.testing.assert_array_equal(pf_grouped2["first"].in_outputs.arr_2d_cs, in_outputs.arr_2d_cs[:, :2])
        assert pf_grouped2["first"].in_outputs.arr_1d_pcg == in_outputs.arr_1d_pcg[0]
        np.testing.assert_array_equal(pf_grouped2["first"].in_outputs.arr_2d_pcg, in_outputs.arr_2d_pcg[:, 0])
        assert pf_grouped2["first"].in_outputs.arr_1d_pg == in_outputs.arr_1d_pg[0]
        np.testing.assert_array_equal(pf_grouped2["first"].in_outputs.arr_2d_pg, in_outputs.arr_2d_pg[:, 0])
        np.testing.assert_array_equal(pf_grouped2["first"].in_outputs.arr_1d_pc, in_outputs.arr_1d_pc[:2])
        np.testing.assert_array_equal(pf_grouped2["first"].in_outputs.arr_2d_pc, in_outputs.arr_2d_pc[:, :2])
        np.testing.assert_array_equal(pf_grouped2["first"].in_outputs.arr_records, in_outputs.arr_records[:10])

        in_outputs = dict(
            arr_1d_cs=np.arange(2),
            arr_2d_cs=np.arange(10).reshape((5, 2)),
            arr_1d_pcg=np.arange(2),
            arr_2d_pcg=np.arange(10).reshape((5, 2)),
            arr_1d_pg=np.arange(2),
            arr_2d_pg=np.arange(10).reshape((5, 2)),
            arr_1d_pc=np.arange(3),
            arr_2d_pc=np.arange(15).reshape((5, 3)),
            arr_1d=np.arange(2),
            arr_2d=np.arange(10).reshape((5, 2)),
            arr_records=np.concatenate(
                (
                    np.full(5, 0, dtype=np.dtype([("col", np.int_)])),
                    np.full(5, 1, dtype=np.dtype([("col", np.int_)])),
                    np.full(5, 2, dtype=np.dtype([("col", np.int_)])),
                )
            ),
        )
        in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        pf_shared2 = pf_shared.replace(in_outputs=in_outputs)

        pd.testing.assert_series_equal(
            pf_shared2.get_in_output("arr_1d_cs"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_cs, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(
            pf_shared2.get_in_output("arr_2d_cs"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_cs),
        )
        pd.testing.assert_series_equal(
            pf_shared2.get_in_output("arr_1d_pcg"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(
            pf_shared2.get_in_output("arr_2d_pcg"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_pcg),
        )
        pd.testing.assert_series_equal(
            pf_shared2.get_in_output("arr_1d_pg"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_pg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(
            pf_shared2.get_in_output("arr_2d_pg"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_pg),
        )
        pd.testing.assert_series_equal(
            pf_shared2.get_in_output("arr_1d_pc"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_pc, name_or_index="arr", group_by=False),
        )
        pd.testing.assert_frame_equal(
            pf_shared2.get_in_output("arr_2d_pc"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_pc, group_by=False),
        )
        pd.testing.assert_series_equal(
            pf_shared2.get_in_output("arr_1d"),
            pf_shared2.wrapper.wrap_reduced(pf_shared2.in_outputs.arr_1d_pcg, name_or_index="arr"),
        )
        pd.testing.assert_frame_equal(
            pf_shared2.get_in_output("arr_2d"),
            pf_shared2.wrapper.wrap(pf_shared2.in_outputs.arr_2d_pcg),
        )
        assert_records_close(pf_shared2.get_in_output("arr_records"), pf_shared2.in_outputs.arr_records)
        with pytest.raises(NotImplementedError):
            pf_shared2.get_in_output("arr_records", force_wrapping=True)

        assert pf_shared2["first"].in_outputs.arr_1d_cs == in_outputs.arr_1d_cs[0]
        np.testing.assert_array_equal(pf_shared2["first"].in_outputs.arr_2d_cs, in_outputs.arr_2d_cs[:, 0])
        assert pf_shared2["first"].in_outputs.arr_1d_pcg == in_outputs.arr_1d_pcg[0]
        np.testing.assert_array_equal(pf_shared2["first"].in_outputs.arr_2d_pcg, in_outputs.arr_2d_pcg[:, 0])
        assert pf_shared2["first"].in_outputs.arr_1d_pg == in_outputs.arr_1d_pg[0]
        np.testing.assert_array_equal(pf_shared2["first"].in_outputs.arr_2d_pg, in_outputs.arr_2d_pg[:, 0])
        np.testing.assert_array_equal(pf_shared2["first"].in_outputs.arr_1d_pc, in_outputs.arr_1d_pc[:2])
        np.testing.assert_array_equal(pf_shared2["first"].in_outputs.arr_2d_pc, in_outputs.arr_2d_pc[:, :2])
        np.testing.assert_array_equal(pf_shared2["first"].in_outputs.arr_records, in_outputs.arr_records[:10])

    def test_close(self):
        pd.testing.assert_frame_equal(pf.close, price_na)
        pd.testing.assert_frame_equal(pf_grouped.close, price_na)
        pd.testing.assert_frame_equal(pf_shared.close, price_na)

    def test_get_filled_close(self):
        pd.testing.assert_frame_equal(pf.filled_close, price_na.ffill().bfill())
        pd.testing.assert_frame_equal(
            pf.filled_close,
            vbt.Portfolio.get_filled_close(close=pf.close, wrapper=pf.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf.get_filled_close(jitted=dict(parallel=True)),
            pf.get_filled_close(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_filled_close(chunked=True), pf.get_filled_close(chunked=False))

    def test_bm_close(self):
        pd.testing.assert_frame_equal(pf.bm_close, bm_price_na)
        pd.testing.assert_frame_equal(pf_grouped.bm_close, bm_price_na)
        pd.testing.assert_frame_equal(pf_shared.bm_close, bm_price_na)
        assert pf.replace(bm_close=None).bm_close is None
        assert not pf.replace(bm_close=False).bm_close

    def test_get_filled_bm_close(self):
        pd.testing.assert_frame_equal(pf.filled_bm_close, bm_price_na.ffill().bfill())
        pd.testing.assert_frame_equal(
            pf.filled_bm_close,
            vbt.Portfolio.get_filled_bm_close(bm_close=pf.bm_close, wrapper=pf.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf.get_filled_bm_close(jitted=dict(parallel=True)),
            pf.get_filled_bm_close(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_filled_bm_close(chunked=True), pf.get_filled_bm_close(chunked=False))
        assert pf.replace(bm_close=None).filled_bm_close is None
        assert not pf.replace(bm_close=False).filled_bm_close

    def test_orders(self):
        result = np.array(
            [
                (0, 0, 1, 0.1, 2.02, 0.10202, 0),
                (1, 0, 2, 1.0, 2.9699999999999998, 0.1297, 1),
                (2, 0, 3, 0.1, 3.96, 0.10396000000000001, 1),
                (3, 0, 4, 1.0, 5.05, 0.1505, 0),
                (0, 1, 1, 0.1, 1.98, 0.10198, 1),
                (1, 1, 3, 0.1, 4.04, 0.10404000000000001, 0),
                (2, 1, 4, 1.0, 4.95, 0.14950000000000002, 1),
                (0, 2, 0, 1.0, 1.01, 0.1101, 0),
                (1, 2, 1, 0.1, 2.02, 0.10202, 0),
                (2, 2, 2, 1.0, 2.9699999999999998, 0.1297, 1),
                (3, 2, 3, 0.1, 3.96, 0.10396000000000001, 1),
            ],
            dtype=order_dt,
        )
        assert_records_close(pf.orders.values, result)
        assert_records_close(pf_grouped.orders.values, result)
        assert_records_close(pf_shared.orders.values, result)
        result2 = pd.Series(np.array([4, 3, 4]), index=price_na.columns).rename("count")
        pd.testing.assert_series_equal(pf.orders.count(), result2)
        pd.testing.assert_series_equal(pf_grouped.get_orders(group_by=False).count(), result2)
        pd.testing.assert_series_equal(pf_shared.get_orders(group_by=False).count(), result2)
        result3 = pd.Series(np.array([7, 4]), index=pd.Index(["first", "second"], dtype="object", name="group")).rename(
            "count"
        )
        pd.testing.assert_series_equal(pf.get_orders(group_by=group_by).count(), result3)
        pd.testing.assert_series_equal(pf_grouped.orders.count(), result3)
        pd.testing.assert_series_equal(pf_shared.orders.count(), result3)

    def test_logs(self):
        result = np.array(
            [
                (
                    0,
                    0,
                    0,
                    0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    100.0,
                    1.0,
                    0.0,
                    100.0,
                    np.nan,
                    np.nan,
                    0.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    100.0,
                    1.0,
                    0.0,
                    100.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    -1,
                ),
                (
                    1,
                    0,
                    0,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.0,
                    100.0,
                    1.0,
                    0.0,
                    100.0,
                    2.0,
                    102.0,
                    0.1,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    99.69598,
                    1.1,
                    0.0,
                    99.69598,
                    2.0,
                    102.0,
                    0.1,
                    2.02,
                    0.10202,
                    0,
                    0,
                    -1,
                    0,
                ),
                (
                    2,
                    0,
                    0,
                    2,
                    np.nan,
                    np.nan,
                    np.nan,
                    3.0,
                    199.69598000000002,
                    1.1,
                    0.0,
                    199.69598000000002,
                    3.0,
                    202.99598000000003,
                    -1.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    202.53628000000003,
                    0.10000000000000009,
                    0.0,
                    202.53628000000003,
                    3.0,
                    202.99598000000003,
                    1.0,
                    2.9699999999999998,
                    0.1297,
                    1,
                    0,
                    -1,
                    1,
                ),
                (
                    3,
                    0,
                    0,
                    3,
                    np.nan,
                    np.nan,
                    np.nan,
                    4.0,
                    202.53628000000003,
                    0.10000000000000009,
                    0.0,
                    202.53628000000003,
                    4.0,
                    202.93628000000004,
                    -0.1,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    202.82832000000002,
                    0.0,
                    0.0,
                    202.82832000000002,
                    4.0,
                    202.93628000000004,
                    0.1,
                    3.96,
                    0.10396000000000001,
                    1,
                    0,
                    -1,
                    2,
                ),
                (
                    4,
                    0,
                    0,
                    4,
                    np.nan,
                    np.nan,
                    np.nan,
                    5.0,
                    202.82832000000002,
                    0.0,
                    0.0,
                    202.82832000000002,
                    5.0,
                    202.82832000000002,
                    1.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    197.62782,
                    1.0,
                    0.0,
                    197.62782,
                    5.0,
                    202.82832000000002,
                    1.0,
                    5.05,
                    0.1505,
                    0,
                    0,
                    -1,
                    3,
                ),
                (
                    0,
                    1,
                    1,
                    0,
                    np.nan,
                    np.nan,
                    np.nan,
                    1.0,
                    100.0,
                    -1.0,
                    0.0,
                    100.0,
                    1.0,
                    99.0,
                    0.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    100.0,
                    -1.0,
                    0.0,
                    100.0,
                    1.0,
                    99.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    5,
                    -1,
                ),
                (
                    1,
                    1,
                    1,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.0,
                    100.0,
                    -1.0,
                    0.0,
                    100.0,
                    2.0,
                    98.0,
                    0.1,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    100.09602,
                    -1.1,
                    0.198,
                    99.70002,
                    2.0,
                    98.0,
                    0.1,
                    1.98,
                    0.10198,
                    1,
                    0,
                    -1,
                    0,
                ),
                (
                    2,
                    1,
                    1,
                    2,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    200.09602,
                    -1.1,
                    0.198,
                    199.70002,
                    2.0,
                    197.89602000000002,
                    -1.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    200.09602,
                    -1.1,
                    0.198,
                    199.70002,
                    2.0,
                    197.89602000000002,
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    -1,
                ),
                (
                    3,
                    1,
                    1,
                    3,
                    np.nan,
                    np.nan,
                    np.nan,
                    4.0,
                    200.09602,
                    -1.1,
                    0.198,
                    199.70002,
                    4.0,
                    195.69602,
                    -0.1,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    199.58798000000002,
                    -1.0,
                    0.18000000000000002,
                    199.22798,
                    4.0,
                    195.69602,
                    0.1,
                    4.04,
                    0.10404000000000001,
                    0,
                    0,
                    -1,
                    1,
                ),
                (
                    4,
                    1,
                    1,
                    4,
                    np.nan,
                    np.nan,
                    np.nan,
                    5.0,
                    199.58798000000002,
                    -1.0,
                    0.18000000000000002,
                    199.22798,
                    5.0,
                    194.58798000000002,
                    1.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    204.38848000000002,
                    -2.0,
                    5.13,
                    194.12848,
                    5.0,
                    194.58798000000002,
                    1.0,
                    4.95,
                    0.14950000000000002,
                    1,
                    0,
                    -1,
                    2,
                ),
                (
                    0,
                    2,
                    2,
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
                    1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    98.8799,
                    1.0,
                    0.0,
                    98.8799,
                    1.0,
                    100.0,
                    1.0,
                    1.01,
                    0.1101,
                    0,
                    0,
                    -1,
                    0,
                ),
                (
                    1,
                    2,
                    2,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.0,
                    98.8799,
                    1.0,
                    0.0,
                    98.8799,
                    2.0,
                    100.8799,
                    0.1,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    98.57588000000001,
                    1.1,
                    0.0,
                    98.57588000000001,
                    2.0,
                    100.8799,
                    0.1,
                    2.02,
                    0.10202,
                    0,
                    0,
                    -1,
                    1,
                ),
                (
                    2,
                    2,
                    2,
                    2,
                    np.nan,
                    np.nan,
                    np.nan,
                    3.0,
                    98.57588000000001,
                    1.1,
                    0.0,
                    98.57588000000001,
                    3.0,
                    101.87588000000001,
                    -1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    101.41618000000001,
                    0.10000000000000009,
                    0.0,
                    101.41618000000001,
                    3.0,
                    101.87588000000001,
                    1.0,
                    2.9699999999999998,
                    0.1297,
                    1,
                    0,
                    -1,
                    2,
                ),
                (
                    3,
                    2,
                    2,
                    3,
                    np.nan,
                    np.nan,
                    np.nan,
                    4.0,
                    101.41618000000001,
                    0.10000000000000009,
                    0.0,
                    101.41618000000001,
                    4.0,
                    101.81618000000002,
                    -0.1,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    101.70822000000001,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.81618000000002,
                    0.1,
                    3.96,
                    0.10396000000000001,
                    1,
                    0,
                    -1,
                    3,
                ),
                (
                    4,
                    2,
                    2,
                    4,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    101.70822000000001,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.70822000000001,
                    1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    101.70822000000001,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.70822000000001,
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    -1,
                ),
            ],
            dtype=log_dt,
        )
        assert_records_close(pf.logs.values, result)
        assert_records_close(pf_grouped.logs.values, result)
        result_shared = np.array(
            [
                (
                    0,
                    0,
                    0,
                    0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    200.0,
                    1.0,
                    0.0,
                    200.0,
                    np.nan,
                    np.nan,
                    0.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    200.0,
                    1.0,
                    0.0,
                    200.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    -1,
                ),
                (
                    1,
                    0,
                    0,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.0,
                    200.09602,
                    1.0,
                    0.0,
                    199.70002,
                    2.0,
                    200.0,
                    0.1,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    199.792,
                    1.1,
                    0.0,
                    199.396,
                    2.0,
                    200.0,
                    0.1,
                    2.02,
                    0.10202,
                    0,
                    0,
                    -1,
                    0,
                ),
                (
                    2,
                    0,
                    0,
                    2,
                    np.nan,
                    np.nan,
                    np.nan,
                    3.0,
                    399.79200000000003,
                    1.1,
                    0.0,
                    399.39599999999996,
                    3.0,
                    400.89200000000005,
                    -1.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    402.63230000000004,
                    0.10000000000000009,
                    0.0,
                    402.23629999999997,
                    3.0,
                    400.89200000000005,
                    1.0,
                    2.9699999999999998,
                    0.1297,
                    1,
                    0,
                    -1,
                    1,
                ),
                (
                    3,
                    0,
                    0,
                    3,
                    np.nan,
                    np.nan,
                    np.nan,
                    4.0,
                    402.12426000000005,
                    0.10000000000000009,
                    0.0,
                    401.76426,
                    4.0,
                    398.63230000000004,
                    -0.1,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    402.41630000000004,
                    0.0,
                    0.0,
                    402.05629999999996,
                    4.0,
                    398.63230000000004,
                    0.1,
                    3.96,
                    0.10396000000000001,
                    1,
                    0,
                    -1,
                    2,
                ),
                (
                    4,
                    0,
                    0,
                    4,
                    np.nan,
                    np.nan,
                    np.nan,
                    5.0,
                    407.21680000000003,
                    0.0,
                    0.0,
                    396.9568,
                    5.0,
                    397.41630000000004,
                    1.0,
                    np.inf,
                    0,
                    0,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    402.01630000000006,
                    1.0,
                    0.0,
                    391.7563,
                    5.0,
                    397.41630000000004,
                    1.0,
                    5.05,
                    0.1505,
                    0,
                    0,
                    -1,
                    3,
                ),
                (
                    0,
                    0,
                    1,
                    0,
                    np.nan,
                    np.nan,
                    np.nan,
                    1.0,
                    200.0,
                    -1.0,
                    0.0,
                    200.0,
                    1.0,
                    np.nan,
                    0.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    200.0,
                    -1.0,
                    0.0,
                    200.0,
                    1.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    5,
                    -1,
                ),
                (
                    1,
                    0,
                    1,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.0,
                    200.0,
                    -1.0,
                    0.0,
                    200.0,
                    2.0,
                    200.0,
                    0.1,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    200.09602,
                    -1.1,
                    0.198,
                    199.70002,
                    2.0,
                    200.0,
                    0.1,
                    1.98,
                    0.10198,
                    1,
                    0,
                    -1,
                    0,
                ),
                (
                    2,
                    0,
                    1,
                    2,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    399.79200000000003,
                    -1.1,
                    0.198,
                    399.39599999999996,
                    2.0,
                    400.89200000000005,
                    -1.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    399.79200000000003,
                    -1.1,
                    0.198,
                    399.39599999999996,
                    2.0,
                    400.89200000000005,
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    -1,
                ),
                (
                    3,
                    0,
                    1,
                    3,
                    np.nan,
                    np.nan,
                    np.nan,
                    4.0,
                    402.63230000000004,
                    -1.1,
                    0.198,
                    402.23629999999997,
                    4.0,
                    398.63230000000004,
                    -0.1,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    402.12426000000005,
                    -1.0,
                    0.18000000000000002,
                    401.76426,
                    4.0,
                    398.63230000000004,
                    0.1,
                    4.04,
                    0.10404000000000001,
                    0,
                    0,
                    -1,
                    1,
                ),
                (
                    4,
                    0,
                    1,
                    4,
                    np.nan,
                    np.nan,
                    np.nan,
                    5.0,
                    402.41630000000004,
                    -1.0,
                    0.18000000000000002,
                    402.05629999999996,
                    5.0,
                    397.41630000000004,
                    1.0,
                    np.inf,
                    0,
                    1,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    407.21680000000003,
                    -2.0,
                    5.13,
                    396.9568,
                    5.0,
                    397.41630000000004,
                    1.0,
                    4.95,
                    0.14950000000000002,
                    1,
                    0,
                    -1,
                    2,
                ),
                (
                    0,
                    1,
                    2,
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
                    1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    98.8799,
                    1.0,
                    0.0,
                    98.8799,
                    1.0,
                    100.0,
                    1.0,
                    1.01,
                    0.1101,
                    0,
                    0,
                    -1,
                    0,
                ),
                (
                    1,
                    1,
                    2,
                    1,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.0,
                    98.8799,
                    1.0,
                    0.0,
                    98.8799,
                    2.0,
                    100.8799,
                    0.1,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    98.57588000000001,
                    1.1,
                    0.0,
                    98.57588000000001,
                    2.0,
                    100.8799,
                    0.1,
                    2.02,
                    0.10202,
                    0,
                    0,
                    -1,
                    1,
                ),
                (
                    2,
                    1,
                    2,
                    2,
                    np.nan,
                    np.nan,
                    np.nan,
                    3.0,
                    98.57588000000001,
                    1.1,
                    0.0,
                    98.57588000000001,
                    3.0,
                    101.87588000000001,
                    -1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    101.41618000000001,
                    0.10000000000000009,
                    0.0,
                    101.41618000000001,
                    3.0,
                    101.87588000000001,
                    1.0,
                    2.9699999999999998,
                    0.1297,
                    1,
                    0,
                    -1,
                    2,
                ),
                (
                    3,
                    1,
                    2,
                    3,
                    np.nan,
                    np.nan,
                    np.nan,
                    4.0,
                    101.41618000000001,
                    0.10000000000000009,
                    0.0,
                    101.41618000000001,
                    4.0,
                    101.81618000000002,
                    -0.1,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    101.70822000000001,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.81618000000002,
                    0.1,
                    3.96,
                    0.10396000000000001,
                    1,
                    0,
                    -1,
                    3,
                ),
                (
                    4,
                    1,
                    2,
                    4,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    101.70822000000001,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.70822000000001,
                    1.0,
                    np.inf,
                    0,
                    2,
                    0.01,
                    0.1,
                    0.01,
                    1e-08,
                    np.inf,
                    np.nan,
                    0.0,
                    0,
                    False,
                    True,
                    False,
                    True,
                    101.70822000000001,
                    0.0,
                    0.0,
                    101.70822000000001,
                    4.0,
                    101.70822000000001,
                    np.nan,
                    np.nan,
                    np.nan,
                    -1,
                    1,
                    1,
                    -1,
                ),
            ],
            dtype=log_dt,
        )
        assert_records_close(pf_shared.logs.values, result_shared)
        result2 = pd.Series(np.array([5, 5, 5]), index=price_na.columns).rename("count")
        pd.testing.assert_series_equal(pf.logs.count(), result2)
        pd.testing.assert_series_equal(pf_grouped.get_logs(group_by=False).count(), result2)
        pd.testing.assert_series_equal(pf_shared.get_logs(group_by=False).count(), result2)
        result3 = pd.Series(
            np.array([10, 5]),
            index=pd.Index(["first", "second"], dtype="object", name="group"),
        ).rename("count")
        pd.testing.assert_series_equal(pf.get_logs(group_by=group_by).count(), result3)
        pd.testing.assert_series_equal(pf_grouped.logs.count(), result3)
        pd.testing.assert_series_equal(pf_shared.logs.count(), result3)

    def test_entry_trades(self):
        result = np.array(
            [
                (0, 0, 1.0, -1, 2.0, 0.0, 3, 3.0599999999999996, 0.21241818181818184, 0.84758182, 0.42379091, 0, 1, 0),
                (
                    1,
                    0,
                    0.1,
                    1,
                    2.02,
                    0.10202,
                    3,
                    3.0599999999999996,
                    0.021241818181818185,
                    -0.019261818181818203,
                    -0.09535553555355546,
                    0,
                    1,
                    0,
                ),
                (2, 0, 1.0, 4, 5.05, 0.1505, 4, 5.0, 0.0, -0.20049999999999982, -0.03970297029702967, 0, 0, 1),
                (
                    0,
                    1,
                    -1.0,
                    -1,
                    1.0,
                    0.0,
                    4,
                    4.954285714285714,
                    -0.049542857142857145,
                    4.003828571428571,
                    -4.003828571428571,
                    1,
                    0,
                    0,
                ),
                (
                    1,
                    1,
                    0.1,
                    1,
                    1.98,
                    0.10198,
                    4,
                    4.954285714285714,
                    0.004954285714285714,
                    -0.4043628571428571,
                    -2.0422366522366517,
                    1,
                    0,
                    0,
                ),
                (
                    2,
                    1,
                    1.0,
                    4,
                    4.95,
                    0.14950000000000002,
                    4,
                    4.954285714285714,
                    0.049542857142857145,
                    -0.20332857142857072,
                    -0.04107647907647893,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    1.0,
                    0,
                    1.01,
                    0.1101,
                    3,
                    3.0599999999999996,
                    0.21241818181818184,
                    1.727481818181818,
                    1.71037803780378,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    2,
                    0.1,
                    1,
                    2.02,
                    0.10202,
                    3,
                    3.0599999999999996,
                    0.021241818181818185,
                    -0.019261818181818203,
                    -0.09535553555355546,
                    0,
                    1,
                    0,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.entry_trades.values, result)
        assert_records_close(pf_grouped.entry_trades.values, result)
        assert_records_close(pf_shared.entry_trades.values, result)
        result2 = pd.Series(np.array([3, 3, 2]), index=price_na.columns).rename("count")
        pd.testing.assert_series_equal(pf.entry_trades.count(), result2)
        pd.testing.assert_series_equal(pf_grouped.get_entry_trades(group_by=False).count(), result2)
        pd.testing.assert_series_equal(pf_shared.get_entry_trades(group_by=False).count(), result2)
        result3 = pd.Series(np.array([6, 2]), index=pd.Index(["first", "second"], dtype="object", name="group")).rename(
            "count"
        )
        pd.testing.assert_series_equal(pf.get_entry_trades(group_by=group_by).count(), result3)
        pd.testing.assert_series_equal(pf_grouped.entry_trades.count(), result3)
        pd.testing.assert_series_equal(pf_shared.entry_trades.count(), result3)

    def test_exit_trades(self):
        result = np.array(
            [
                (
                    0,
                    0,
                    1.0,
                    -1,
                    2.0018181818181815,
                    0.09274545454545455,
                    2,
                    2.9699999999999998,
                    0.1297,
                    0.7457363636363636,
                    0.37252951861943695,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    0,
                    0.10000000000000009,
                    -1,
                    2.0018181818181815,
                    0.009274545454545462,
                    3,
                    3.96,
                    0.10396000000000001,
                    0.08258363636363657,
                    0.41254314259763925,
                    0,
                    1,
                    0,
                ),
                (2, 0, 1.0, 4, 5.05, 0.1505, 4, 5.0, 0.0, -0.20049999999999982, -0.03970297029702967, 0, 0, 1),
                (
                    0,
                    1,
                    0.1,
                    -1,
                    1.0890909090909089,
                    0.009270909090909092,
                    3,
                    4.04,
                    0.10404000000000001,
                    -0.40840181818181825,
                    -3.749933222036729,
                    1,
                    1,
                    0,
                ),
                (
                    1,
                    1,
                    2.0,
                    -1,
                    3.0195454545454545,
                    0.24220909090909093,
                    4,
                    5.0,
                    0.0,
                    -4.203118181818182,
                    -0.6959852476290832,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    1.0,
                    0,
                    1.1018181818181818,
                    0.19283636363636364,
                    2,
                    2.9699999999999998,
                    0.1297,
                    1.5456454545454543,
                    1.4028135313531351,
                    0,
                    1,
                    0,
                ),
                (
                    1,
                    2,
                    0.10000000000000009,
                    0,
                    1.1018181818181818,
                    0.019283636363636378,
                    3,
                    3.96,
                    0.10396000000000001,
                    0.1625745454545457,
                    1.4755115511551162,
                    0,
                    1,
                    0,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.exit_trades.values, result)
        assert_records_close(pf_grouped.exit_trades.values, result)
        assert_records_close(pf_shared.exit_trades.values, result)
        result2 = pd.Series(np.array([3, 2, 2]), index=price_na.columns).rename("count")
        pd.testing.assert_series_equal(pf.exit_trades.count(), result2)
        pd.testing.assert_series_equal(pf_grouped.get_exit_trades(group_by=False).count(), result2)
        pd.testing.assert_series_equal(pf_shared.get_exit_trades(group_by=False).count(), result2)
        result3 = pd.Series(np.array([5, 2]), index=pd.Index(["first", "second"], dtype="object", name="group")).rename(
            "count"
        )
        pd.testing.assert_series_equal(pf.get_exit_trades(group_by=group_by).count(), result3)
        pd.testing.assert_series_equal(pf_grouped.exit_trades.count(), result3)
        pd.testing.assert_series_equal(pf_shared.exit_trades.count(), result3)

    def test_positions(self):
        result = np.array(
            [
                (
                    0,
                    0,
                    1.1,
                    -1,
                    2.001818,
                    0.10202000000000001,
                    3,
                    3.06,
                    0.23366000000000003,
                    0.82832,
                    0.37616712,
                    0,
                    1,
                    0,
                ),
                (1, 0, 1.0, 4, 5.05, 0.1505, 4, 5.0, 0.0, -0.20049999999999982, -0.03970297029702967, 0, 0, 1),
                (
                    0,
                    1,
                    2.1,
                    -1,
                    2.9276190476190473,
                    0.25148000000000004,
                    4,
                    4.954285714285714,
                    0.10404000000000001,
                    -4.6115200000000005,
                    -0.7500845803513339,
                    1,
                    0,
                    0,
                ),
                (
                    0,
                    2,
                    1.1,
                    0,
                    1.1018181818181818,
                    0.21212000000000003,
                    3,
                    3.06,
                    0.23366000000000003,
                    1.7082200000000003,
                    1.4094224422442245,
                    0,
                    1,
                    0,
                ),
            ],
            dtype=trade_dt,
        )
        assert_records_close(pf.positions.values, result)
        assert_records_close(pf_grouped.positions.values, result)
        assert_records_close(pf_shared.positions.values, result)
        result2 = pd.Series(np.array([2, 1, 1]), index=price_na.columns).rename("count")
        pd.testing.assert_series_equal(pf.positions.count(), result2)
        pd.testing.assert_series_equal(pf_grouped.get_positions(group_by=False).count(), result2)
        pd.testing.assert_series_equal(pf_shared.get_positions(group_by=False).count(), result2)
        result3 = pd.Series(np.array([3, 1]), index=pd.Index(["first", "second"], dtype="object", name="group")).rename(
            "count"
        )
        pd.testing.assert_series_equal(pf.get_positions(group_by=group_by).count(), result3)
        pd.testing.assert_series_equal(pf_grouped.positions.count(), result3)
        pd.testing.assert_series_equal(pf_shared.positions.count(), result3)

    def test_drawdowns(self):
        result = np.array(
            [
                (0, 0, 0, 1, 1, 2, 102.0, 101.89598000000001, 202.83628000000004, 1),
                (1, 0, 2, 3, 4, 4, 202.83628000000004, 202.62782, 202.62782, 0),
                (0, 1, 0, 1, 1, 2, 99.0, 97.89602, 197.89602000000002, 1),
                (1, 1, 2, 3, 4, 4, 197.89602000000002, 194.38848000000002, 194.38848000000002, 0),
                (0, 2, 2, 3, 3, 4, 101.71618000000001, 101.70822000000001, 101.70822000000001, 0),
            ],
            dtype=drawdown_dt,
        )
        assert_records_close(pf.drawdowns.values, result)
        result_grouped = np.array(
            [
                (0, 0, 0, 1, 1, 2, 201.0, 199.792, 400.73230000000007, 1),
                (1, 0, 2, 3, 4, 4, 400.73230000000007, 397.01630000000006, 397.01630000000006, 0),
                (0, 1, 2, 3, 3, 4, 101.71618000000001, 101.70822000000001, 101.70822000000001, 0),
            ],
            dtype=drawdown_dt,
        )
        assert_records_close(pf_grouped.drawdowns.values, result_grouped)
        assert_records_close(pf_shared.drawdowns.values, result_grouped)
        result2 = pd.Series(np.array([2, 2, 1]), index=price_na.columns).rename("count")
        pd.testing.assert_series_equal(pf.drawdowns.count(), result2)
        pd.testing.assert_series_equal(pf_grouped.get_drawdowns(group_by=False).count(), result2)
        pd.testing.assert_series_equal(pf_shared.get_drawdowns(group_by=False).count(), result2)
        result3 = pd.Series(np.array([2, 1]), index=pd.Index(["first", "second"], dtype="object", name="group")).rename(
            "count"
        )
        pd.testing.assert_series_equal(pf.get_drawdowns(group_by=group_by).count(), result3)
        pd.testing.assert_series_equal(pf_grouped.drawdowns.count(), result3)
        pd.testing.assert_series_equal(pf_shared.drawdowns.count(), result3)

    def test_init_position(self):
        result = pd.Series(np.array([1.0, -1.0, 0.0]), index=price_na.columns).rename("init_position")
        pd.testing.assert_series_equal(pf.init_position, result)
        pd.testing.assert_series_equal(pf_grouped.init_position, result)
        pd.testing.assert_series_equal(pf_shared.init_position, result)

    def test_asset_flow(self):
        pd.testing.assert_frame_equal(
            pf.get_asset_flow(direction="longonly"),
            pd.DataFrame(
                np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 0.1], [-1, 0.0, -1.0], [-0.1, 0.0, -0.1], [1.0, 0.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_asset_flow(direction="shortonly"),
            pd.DataFrame(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 1.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array([[0.0, 0.0, 1.0], [0.1, -0.1, 0.1], [-1, 0.0, -1.0], [-0.1, 0.1, -0.1], [1.0, -1.0, 0.0]]),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.asset_flow, result)
        pd.testing.assert_frame_equal(pf_grouped.asset_flow, result)
        pd.testing.assert_frame_equal(pf_shared.asset_flow, result)
        pd.testing.assert_frame_equal(
            pf.asset_flow,
            vbt.Portfolio.get_asset_flow(orders=pf.orders, init_position=pf.init_position),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.asset_flow,
            vbt.Portfolio.get_asset_flow(orders=pf_grouped.orders, init_position=pf_grouped.init_position),
        )
        pd.testing.assert_frame_equal(
            pf_shared.asset_flow,
            vbt.Portfolio.get_asset_flow(orders=pf_shared.orders, init_position=pf_shared.init_position),
        )
        pd.testing.assert_frame_equal(
            pf.get_asset_flow(jitted=dict(parallel=True)),
            pf.get_asset_flow(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_asset_flow(chunked=False), pf.get_asset_flow(chunked=True))

    def test_assets(self):
        pd.testing.assert_frame_equal(
            pf.get_assets(direction="longonly"),
            pd.DataFrame(
                np.array([[1.0, 0.0, 1.0], [1.1, 0.0, 1.1], [0.1, 0.0, 0.1], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_assets(direction="shortonly"),
            pd.DataFrame(
                np.array([[0.0, 1.0, 0.0], [0.0, 1.1, 0.0], [0.0, 1.1, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array([[1.0, -1.0, 1.0], [1.1, -1.1, 1.1], [0.1, -1.1, 0.1], [0.0, -1.0, 0.0], [1.0, -2.0, 0.0]]),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.assets, result)
        pd.testing.assert_frame_equal(pf_grouped.assets, result)
        pd.testing.assert_frame_equal(pf_shared.assets, result)
        pd.testing.assert_frame_equal(
            pf.assets,
            vbt.Portfolio.get_assets(asset_flow=pf.asset_flow, init_position=pf.init_position, wrapper=pf.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.assets,
            vbt.Portfolio.get_assets(
                asset_flow=pf_grouped.asset_flow,
                init_position=pf_grouped.init_position,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.assets,
            vbt.Portfolio.get_assets(
                asset_flow=pf_shared.asset_flow,
                init_position=pf_shared.init_position,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_assets(jitted=dict(parallel=True)),
            pf.get_assets(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_assets(chunked=False), pf.get_assets(chunked=True))

    def test_position_mask(self):
        pd.testing.assert_frame_equal(
            pf.get_position_mask(direction="longonly"),
            pd.DataFrame(
                np.array(
                    [
                        [True, False, True],
                        [True, False, True],
                        [True, False, True],
                        [False, False, False],
                        [True, False, False],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_position_mask(direction="shortonly"),
            pd.DataFrame(
                np.array(
                    [
                        [False, True, False],
                        [False, True, False],
                        [False, True, False],
                        [False, True, False],
                        [False, True, False],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [[True, True, True], [True, True, True], [True, True, True], [False, True, False], [True, True, False]],
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.position_mask, result)
        pd.testing.assert_frame_equal(pf_grouped.get_position_mask(group_by=False), result)
        pd.testing.assert_frame_equal(pf_shared.get_position_mask(group_by=False), result)
        result = pd.DataFrame(
            np.array([[True, True], [True, True], [True, True], [True, False], [True, False]]),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_position_mask(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.position_mask, result)
        pd.testing.assert_frame_equal(pf_shared.position_mask, result)
        pd.testing.assert_frame_equal(
            pf.position_mask,
            vbt.Portfolio.get_position_mask(assets=pf.assets, wrapper=pf.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.position_mask,
            vbt.Portfolio.get_position_mask(assets=pf_grouped.assets, wrapper=pf_grouped.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_shared.position_mask,
            vbt.Portfolio.get_position_mask(assets=pf_shared.assets, wrapper=pf_shared.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_position_mask(jitted=dict(parallel=True)),
            pf_grouped.get_position_mask(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_position_mask(chunked=False),
            pf_grouped.get_position_mask(chunked=True),
        )

    def test_position_coverage(self):
        pd.testing.assert_series_equal(
            pf.get_position_coverage(direction="longonly"),
            pd.Series(np.array([0.8, 0.0, 0.6]), index=price_na.columns).rename("position_coverage"),
        )
        pd.testing.assert_series_equal(
            pf.get_position_coverage(direction="shortonly"),
            pd.Series(np.array([0.0, 1.0, 0.0]), index=price_na.columns).rename("position_coverage"),
        )
        result = pd.Series(np.array([0.8, 1.0, 0.6]), index=price_na.columns).rename("position_coverage")
        pd.testing.assert_series_equal(pf.position_coverage, result)
        pd.testing.assert_series_equal(pf_grouped.get_position_coverage(group_by=False), result)
        pd.testing.assert_series_equal(pf_shared.get_position_coverage(group_by=False), result)
        result = pd.Series(np.array([0.9, 0.6]), pd.Index(["first", "second"], dtype="object", name="group")).rename(
            "position_coverage"
        )
        pd.testing.assert_series_equal(pf.get_position_coverage(group_by=group_by), result)
        pd.testing.assert_series_equal(pf_grouped.position_coverage, result)
        pd.testing.assert_series_equal(pf_shared.position_coverage, result)
        pd.testing.assert_series_equal(
            pf.position_coverage,
            vbt.Portfolio.get_position_coverage(position_mask=pf.get_position_mask(group_by=False), wrapper=pf.wrapper),
        )
        pd.testing.assert_series_equal(
            pf_grouped.position_coverage,
            vbt.Portfolio.get_position_coverage(
                position_mask=pf_grouped.get_position_mask(group_by=False),
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.position_coverage,
            vbt.Portfolio.get_position_coverage(
                position_mask=pf_shared.get_position_mask(group_by=False),
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.get_position_coverage(jitted=dict(parallel=True)),
            pf_grouped.get_position_coverage(jitted=dict(parallel=False)),
        )
        pd.testing.assert_series_equal(
            pf_grouped.get_position_coverage(chunked=False),
            pf_grouped.get_position_coverage(chunked=True),
        )

    def test_cash_flow(self):
        pd.testing.assert_frame_equal(
            pf.get_cash_flow(free=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, -1.1201],
                        [-0.30402, -0.29998, -0.3040200000000002],
                        [-2.5057000000000005, 0.0, 2.8402999999999996],
                        [-0.4999599999999999, -0.11204000000000003, 0.29204000000000035],
                        [0.9375, -5.0995, 0.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, -1.1201],
                    [-0.30402, 0.09602000000000001, -0.30402],
                    [2.8402999999999996, 0.0, 2.8402999999999996],
                    [0.29204, -0.50804, 0.29204],
                    [-5.2005, 4.8005, 0.0],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.cash_flow, result)
        pd.testing.assert_frame_equal(pf_grouped.get_cash_flow(group_by=False), result)
        pd.testing.assert_frame_equal(pf_shared.get_cash_flow(group_by=False), result)
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, -1.1201],
                    [-0.20800000000000002, -0.30402],
                    [2.8402999999999996, 2.8402999999999996],
                    [-0.21600000000000003, 0.29204],
                    [-0.39999999999999947, 0.0],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_cash_flow(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.cash_flow, result)
        pd.testing.assert_frame_equal(pf_shared.cash_flow, result)
        pd.testing.assert_frame_equal(pf.cash_flow, vbt.Portfolio.get_cash_flow(orders=pf.orders, wrapper=pf.wrapper))
        pd.testing.assert_frame_equal(
            pf_grouped.cash_flow,
            vbt.Portfolio.get_cash_flow(orders=pf_grouped.orders, wrapper=pf_grouped.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_shared.cash_flow,
            vbt.Portfolio.get_cash_flow(orders=pf_shared.orders, wrapper=pf_shared.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf.get_cash_flow(jitted=dict(parallel=True)),
            pf.get_cash_flow(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_cash_flow(chunked=False), pf.get_cash_flow(chunked=True))
        pd.testing.assert_frame_equal(
            pf_grouped.get_cash_flow(jitted=dict(parallel=True)),
            pf_grouped.get_cash_flow(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf_grouped.get_cash_flow(chunked=False), pf_grouped.get_cash_flow(chunked=True))

    def test_init_cash(self):
        pd.testing.assert_series_equal(
            pf.init_cash,
            pd.Series(np.array([100.0, 100.0, 100.0]), index=price_na.columns).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            pf_grouped.get_init_cash(group_by=False),
            pd.Series(np.array([100.0, 100.0, 100.0]), index=price_na.columns).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_init_cash(group_by=False),
            pd.Series(np.array([200.0, 200.0, 100.0]), index=price_na.columns).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_init_cash(group_by=False, split_shared=True),
            pd.Series(np.array([100.0, 100.0, 100.0]), index=price_na.columns).rename("init_cash"),
        )
        result = pd.Series(
            np.array([200.0, 100.0]),
            pd.Index(["first", "second"], dtype="object", name="group"),
        ).rename("init_cash")
        pd.testing.assert_series_equal(pf.get_init_cash(group_by=group_by), result)
        pd.testing.assert_series_equal(pf_grouped.init_cash, result)
        pd.testing.assert_series_equal(pf_shared.init_cash, result)
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(price_na, 1000.0, init_cash=InitCashMode.Auto, group_by=None).init_cash,
            pd.Series(np.array([14000.0, 12000.0, 10000.0]), index=price_na.columns).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(price_na, 1000.0, init_cash=InitCashMode.Auto, group_by=group_by).init_cash,
            pd.Series(
                np.array([26000.0, 10000.0]),
                index=pd.Index(["first", "second"], dtype="object", name="group"),
            ).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na,
                1000.0,
                init_cash=InitCashMode.Auto,
                group_by=group_by,
                cash_sharing=True,
            ).init_cash,
            pd.Series(
                np.array([26000.0, 10000.0]),
                index=pd.Index(["first", "second"], dtype="object", name="group"),
            ).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(price_na, 1000.0, init_cash=InitCashMode.AutoAlign, group_by=None).init_cash,
            pd.Series(np.array([14000.0, 14000.0, 14000.0]), index=price_na.columns).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(price_na, 1000.0, init_cash=InitCashMode.AutoAlign, group_by=group_by).init_cash,
            pd.Series(
                np.array([26000.0, 26000.0]),
                index=pd.Index(["first", "second"], dtype="object", name="group"),
            ).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na,
                1000.0,
                init_cash=InitCashMode.AutoAlign,
                group_by=group_by,
                cash_sharing=True,
            ).init_cash,
            pd.Series(
                np.array([26000.0, 26000.0]),
                index=pd.Index(["first", "second"], dtype="object", name="group"),
            ).rename("init_cash"),
        )
        pd.testing.assert_series_equal(
            pf.init_cash,
            vbt.Portfolio.get_init_cash(
                init_cash_raw=pf._init_cash,
                cash_sharing=pf.cash_sharing,
                free_cash_flow=pf.free_cash_flow,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.init_cash,
            vbt.Portfolio.get_init_cash(
                init_cash_raw=pf_grouped._init_cash,
                cash_sharing=pf_grouped.cash_sharing,
                free_cash_flow=pf_grouped.free_cash_flow,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.init_cash,
            vbt.Portfolio.get_init_cash(
                init_cash_raw=pf_shared._init_cash,
                cash_sharing=pf_shared.cash_sharing,
                free_cash_flow=pf_shared.free_cash_flow,
                wrapper=pf_shared.wrapper,
            ),
        )
        pf2 = vbt.Portfolio.from_orders(
            price_na,
            1000.0,
            init_cash=InitCashMode.AutoAlign,
            group_by=group_by,
            cash_sharing=True,
        )
        pd.testing.assert_series_equal(
            pf2.init_cash,
            type(pf2).get_init_cash(
                init_cash_raw=pf2._init_cash,
                cash_sharing=pf2.cash_sharing,
                free_cash_flow=pf2.free_cash_flow,
                wrapper=pf2.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(price_na, 1000.0, init_cash=InitCashMode.Auto).get_init_cash(
                jitted=dict(parallel=True)
            ),
            vbt.Portfolio.from_orders(price_na, 1000.0, init_cash=InitCashMode.Auto).get_init_cash(
                jitted=dict(parallel=False)
            ),
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(price_na, 1000.0, init_cash=InitCashMode.Auto).get_init_cash(chunked=True),
            vbt.Portfolio.from_orders(price_na, 1000.0, init_cash=InitCashMode.Auto).get_init_cash(chunked=False),
        )

    def test_cash_deposits(self):
        pd.testing.assert_frame_equal(
            pf.cash_deposits,
            pd.DataFrame(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_cash_deposits(group_by=False),
            pd.DataFrame(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.get_cash_deposits(group_by=False),
            pd.DataFrame(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [200.0, 200.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.get_cash_deposits(group_by=False, split_shared=True),
            pd.DataFrame(
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [100.0, 100.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array([[0.0, 0.0], [0.0, 0.0], [200.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_cash_deposits(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.cash_deposits, result)
        pd.testing.assert_frame_equal(pf_shared.cash_deposits, result)
        pd.testing.assert_frame_equal(
            pf.cash_deposits,
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=pf._cash_deposits,
                cash_sharing=pf.cash_sharing,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.cash_deposits,
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=pf_grouped._cash_deposits,
                cash_sharing=pf_grouped.cash_sharing,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.cash_deposits,
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=pf_shared._cash_deposits,
                cash_sharing=pf_shared.cash_sharing,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_cash_deposits(jitted=dict(parallel=True)),
            pf.get_cash_deposits(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_cash_deposits(chunked=True), pf.get_cash_deposits(chunked=False))
        pd.testing.assert_frame_equal(
            pf_grouped.get_cash_deposits(jitted=dict(parallel=True)),
            pf_grouped.get_cash_deposits(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_cash_deposits(chunked=True),
            pf_grouped.get_cash_deposits(chunked=False),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=1,
                keep_flex=True,
                cash_sharing=True,
                wrapper=pf_grouped.wrapper,
            ),
            np.array([[1]]),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=1,
                keep_flex=True,
                cash_sharing=False,
                wrapper=pf_grouped.wrapper,
            ),
            np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_deposits(
                cash_deposits_raw=1,
                keep_flex=True,
                cash_sharing=False,
                wrapper=pf.wrapper,
            ),
            np.array([[1]]),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_deposits(cash_deposits_raw=1, keep_flex=True, cash_sharing=True, wrapper=pf.wrapper),
            np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        )

    def test_cash_earnings(self):
        result = pd.DataFrame(
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.cash_earnings, result)
        pd.testing.assert_frame_equal(pf_grouped.get_cash_earnings(group_by=False), result)
        pd.testing.assert_frame_equal(pf_shared.get_cash_earnings(group_by=False), result)
        result = pd.DataFrame(
            np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_cash_earnings(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.cash_earnings, result)
        pd.testing.assert_frame_equal(pf_shared.cash_earnings, result)
        pd.testing.assert_frame_equal(
            pf.cash_earnings,
            vbt.Portfolio.get_cash_earnings(cash_earnings_raw=pf._cash_earnings, wrapper=pf.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.cash_earnings,
            vbt.Portfolio.get_cash_earnings(cash_earnings_raw=pf_grouped._cash_earnings, wrapper=pf_grouped.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_shared.cash_earnings,
            vbt.Portfolio.get_cash_earnings(cash_earnings_raw=pf_shared._cash_earnings, wrapper=pf_shared.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_cash_earnings(jitted=dict(parallel=True)),
            pf_grouped.get_cash_earnings(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_cash_earnings(chunked=True),
            pf_grouped.get_cash_earnings(chunked=False),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_earnings(cash_earnings_raw=1, keep_flex=True, wrapper=pf.wrapper),
            np.array([[1]]),
        )
        np.testing.assert_array_equal(
            vbt.Portfolio.get_cash_earnings(cash_earnings_raw=1, keep_flex=True, wrapper=pf_grouped.wrapper),
            np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]]),
        )

    def test_cash(self):
        pd.testing.assert_frame_equal(
            pf.get_cash(free=True),
            pd.DataFrame(
                np.array(
                    [
                        [100.0, 100.0, 98.8799],
                        [99.69598, 99.70002, 98.57588000000001],
                        [197.19028000000003, 199.70002, 101.41618000000001],
                        [196.69032000000004, 199.58798, 101.70822000000001],
                        [197.62782000000004, 194.48847999999998, 101.70822000000001],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [100.0, 100.0, 98.8799],
                    [99.69598, 100.09602, 98.57588000000001],
                    [202.53628000000003, 200.09602, 101.41618000000001],
                    [202.82832000000002, 199.58798000000002, 101.70822000000001],
                    [197.62782, 204.38848000000002, 101.70822000000001],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.cash, result)
        pd.testing.assert_frame_equal(pf_grouped.get_cash(group_by=False), result)
        pd.testing.assert_frame_equal(
            pf_shared.get_cash(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [200.0, 200.0, 98.8799],
                        [199.69598, 200.09602, 98.57588000000001],
                        [402.53628, 400.09602, 101.41618000000001],
                        [402.82831999999996, 399.58798, 101.70822000000001],
                        [397.62782, 404.38848, 101.70822000000001],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [200.0, 98.8799],
                    [199.792, 98.57588000000001],
                    [402.63230000000004, 101.41618000000001],
                    [402.41630000000004, 101.70822000000001],
                    [402.01630000000006, 101.70822000000001],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_cash(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.cash, result)
        pd.testing.assert_frame_equal(pf_shared.cash, result)
        pd.testing.assert_frame_equal(
            pf.cash,
            vbt.Portfolio.get_cash(
                init_cash=pf.init_cash,
                cash_deposits=pf.cash_deposits,
                cash_sharing=pf.cash_sharing,
                cash_flow=pf.cash_flow,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.cash,
            vbt.Portfolio.get_cash(
                init_cash=pf_grouped.init_cash,
                cash_deposits=pf_grouped.cash_deposits,
                cash_sharing=pf_grouped.cash_sharing,
                cash_flow=pf_grouped.cash_flow,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(pf.get_cash(jitted=dict(parallel=True)), pf.get_cash(jitted=dict(parallel=False)))
        pd.testing.assert_frame_equal(pf.get_cash(chunked=True), pf.get_cash(chunked=False))
        pd.testing.assert_frame_equal(
            pf_grouped.get_cash(jitted=dict(parallel=True)),
            pf_grouped.get_cash(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf_grouped.get_cash(chunked=True), pf_grouped.get_cash(chunked=False))

    def test_init_position_value(self):
        result = pd.Series(np.array([2.0, -1.0, 0.0]), index=price_na.columns).rename("init_position_value")
        pd.testing.assert_series_equal(pf.init_position_value, result)
        pd.testing.assert_series_equal(pf_grouped.init_position_value, result)
        pd.testing.assert_series_equal(pf_shared.init_position_value, result)
        pd.testing.assert_series_equal(
            pf.init_position_value,
            vbt.Portfolio.get_init_position_value(
                init_price=pf.init_price,
                init_position=pf.init_position,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.init_position_value,
            vbt.Portfolio.get_init_position_value(
                init_price=pf_grouped.init_price,
                init_position=pf_grouped.init_position,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.init_position_value,
            vbt.Portfolio.get_init_position_value(
                init_price=pf_shared.init_price,
                init_position=pf_shared.init_position,
                wrapper=pf_shared.wrapper,
            ),
        )

    def test_init_value(self):
        pd.testing.assert_series_equal(
            pf.init_value,
            pd.Series(np.array([102.0, 99.0, 100.0]), index=price_na.columns).rename("init_value"),
        )
        pd.testing.assert_series_equal(
            pf_grouped.get_init_value(group_by=False),
            pd.Series(np.array([102.0, 99.0, 100.0]), index=price_na.columns).rename("init_value"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_init_value(group_by=False),
            pd.Series(np.array([202.0, 199.0, 100.0]), index=price_na.columns).rename("init_value"),
        )
        result = pd.Series(
            np.array([201.0, 100.0]),
            pd.Index(["first", "second"], dtype="object", name="group"),
        ).rename("init_value")
        pd.testing.assert_series_equal(pf.get_init_value(group_by=group_by), result)
        pd.testing.assert_series_equal(pf_grouped.init_value, result)
        pd.testing.assert_series_equal(pf_shared.init_value, result)
        pd.testing.assert_series_equal(
            pf.get_init_value(jitted=dict(parallel=True)),
            pf.get_init_value(jitted=dict(parallel=False)),
        )
        pd.testing.assert_series_equal(pf.get_init_value(chunked=True), pf.get_init_value(chunked=False))
        pd.testing.assert_series_equal(
            pf.init_value,
            vbt.Portfolio.get_init_value(
                init_position_value=pf.init_position_value,
                init_cash=pf.init_cash,
                split_shared=False,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf.init_value,
            vbt.Portfolio.get_init_value(
                init_position_value=pf.init_position_value,
                init_cash=pf.init_cash,
                split_shared=True,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.init_value,
            vbt.Portfolio.get_init_value(
                init_position_value=pf_grouped.init_position_value,
                init_cash=pf_grouped.init_cash,
                split_shared=False,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.init_value,
            vbt.Portfolio.get_init_value(
                init_position_value=pf_shared.init_position_value,
                init_cash=pf_shared.init_cash,
                split_shared=False,
                wrapper=pf_shared.wrapper,
            ),
        )

    def test_input_value(self):
        pd.testing.assert_series_equal(
            pf.input_value,
            pd.Series(np.array([202.0, 199.0, 100.0]), index=price_na.columns).rename("input_value"),
        )
        pd.testing.assert_series_equal(
            pf_grouped.get_input_value(group_by=False),
            pd.Series(np.array([202.0, 199.0, 100.0]), index=price_na.columns).rename("input_value"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_input_value(group_by=False),
            pd.Series(np.array([402.0, 399.0, 100.0]), index=price_na.columns).rename("input_value"),
        )
        result = pd.Series(
            np.array([401.0, 100.0]),
            pd.Index(["first", "second"], dtype="object", name="group"),
        ).rename("input_value")
        pd.testing.assert_series_equal(pf.get_input_value(group_by=group_by), result)
        pd.testing.assert_series_equal(pf_grouped.input_value, result)
        pd.testing.assert_series_equal(pf_shared.input_value, result)
        pd.testing.assert_series_equal(
            pf.get_input_value(jitted=dict(parallel=True)),
            pf.get_input_value(jitted=dict(parallel=False)),
        )
        pd.testing.assert_series_equal(pf.get_input_value(chunked=True), pf.get_input_value(chunked=False))
        pd.testing.assert_series_equal(
            pf.input_value,
            vbt.Portfolio.get_input_value(
                init_value=pf.init_value,
                cash_deposits_raw=pf._cash_deposits,
                cash_sharing=pf.cash_sharing,
                split_shared=False,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf.input_value,
            vbt.Portfolio.get_input_value(
                init_value=pf.init_value,
                cash_deposits_raw=pf._cash_deposits,
                cash_sharing=pf.cash_sharing,
                split_shared=True,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.input_value,
            vbt.Portfolio.get_input_value(
                init_value=pf_grouped.init_value,
                cash_deposits_raw=pf_grouped._cash_deposits,
                cash_sharing=pf_grouped.cash_sharing,
                split_shared=False,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.input_value,
            vbt.Portfolio.get_input_value(
                init_value=pf_shared.init_value,
                cash_deposits_raw=pf_shared._cash_deposits,
                cash_sharing=pf_shared.cash_sharing,
                split_shared=False,
                wrapper=pf_shared.wrapper,
            ),
        )

    def test_asset_value(self):
        pd.testing.assert_frame_equal(
            pf.get_asset_value(direction="longonly"),
            pd.DataFrame(
                np.array(
                    [
                        [2.0, 0.0, 1.0],
                        [2.2, 0.0, 2.2],
                        [0.30000000000000027, 0.0, 0.30000000000000027],
                        [0.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_asset_value(direction="shortonly"),
            pd.DataFrame(
                np.array([[0.0, 1.0, 0.0], [0.0, 2.2, 0.0], [0.0, 2.2, 0.0], [0.0, 4.0, 0.0], [0.0, 10.0, 0.0]]),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [2.0, -1.0, 1.0],
                    [2.2, -2.2, 2.2],
                    [0.30000000000000027, -2.2, 0.30000000000000027],
                    [0.0, -4.0, 0.0],
                    [5.0, -10.0, 0.0],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.asset_value, result)
        pd.testing.assert_frame_equal(pf_grouped.get_asset_value(group_by=False), result)
        pd.testing.assert_frame_equal(pf_shared.get_asset_value(group_by=False), result)
        result = pd.DataFrame(
            np.array([[1.0, 1.0], [0.0, 2.2], [-1.9, 0.30000000000000027], [-4.0, 0.0], [-5.0, 0.0]]),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_asset_value(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.asset_value, result)
        pd.testing.assert_frame_equal(pf_shared.asset_value, result)
        pd.testing.assert_frame_equal(
            pf.asset_value,
            vbt.Portfolio.get_asset_value(close=pf.filled_close, assets=pf.assets, wrapper=pf.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.asset_value,
            vbt.Portfolio.get_asset_value(
                close=pf_grouped.filled_close,
                assets=pf_grouped.assets,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.asset_value,
            vbt.Portfolio.get_asset_value(
                close=pf_shared.filled_close,
                assets=pf_shared.assets,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_asset_value(jitted=dict(parallel=True)),
            pf_grouped.get_asset_value(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_asset_value(chunked=True),
            pf_grouped.get_asset_value(chunked=False),
        )

    def test_gross_exposure(self):
        pd.testing.assert_frame_equal(
            pf.get_gross_exposure(direction="longonly"),
            pd.DataFrame(
                np.array(
                    [
                        [0.0196078431372549, 0.0, 0.010012024441354066],
                        [0.021590645676110087, 0.0, 0.021830620581035857],
                        [0.001519062102701967, 0.0, 0.002949383274126105],
                        [0.0, 0.0, 0.0],
                        [0.02467578242711193, 0.0, 0.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_gross_exposure(direction="shortonly"),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.009900990099009901, 0.0],
                        [0.0, 0.02158978967815708, 0.0],
                        [0.0, 0.01089648232823355, 0.0],
                        [0.0, 0.019647525359797767, 0.0],
                        [0.0, 0.04890251030278087, 0.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0196078431372549, -0.010101010101010102, 0.010012024441354066],
                    [0.021590645676110087, -0.022564097935569658, 0.021830620581035857],
                    [0.001519062102701967, -0.011139239378304874, 0.002949383274126105],
                    [0.0, -0.020451154513687397, 0.0],
                    [0.02467578242711193, -0.05420392644570545, 0.0],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.gross_exposure, result)
        pd.testing.assert_frame_equal(pf_grouped.get_gross_exposure(group_by=False), result)
        pd.testing.assert_frame_equal(
            pf_shared.get_gross_exposure(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.009900990099009901, -0.005025125628140704, 0.010012024441354066],
                        [0.010896700370160913, -0.011139239378304874, 0.021830620581035857],
                        [0.0007547354365495435, -0.0055345909164985704, 0.002949383274126105],
                        [0.0, -0.010111530689077055, 0.0],
                        [0.01241841659128274, -0.02600858158351064, 0.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.004975124378109453, 0.010012024441354066],
                    [0.0, 0.021830620581035857],
                    [-0.00481024470727509, 0.002949383274126105],
                    [-0.010196842394799815, 0.0],
                    [-0.012916015161335238, 0.0],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_gross_exposure(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.gross_exposure, result)
        pd.testing.assert_frame_equal(pf_shared.gross_exposure, result)
        pd.testing.assert_frame_equal(
            pf.gross_exposure,
            vbt.Portfolio.get_gross_exposure(
                asset_value=pf.asset_value,
                free_cash=pf.get_cash(free=True),
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.gross_exposure,
            vbt.Portfolio.get_gross_exposure(
                asset_value=pf_grouped.asset_value,
                free_cash=pf_grouped.get_cash(free=True),
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.gross_exposure,
            vbt.Portfolio.get_gross_exposure(
                asset_value=pf_shared.asset_value,
                free_cash=pf_shared.get_cash(free=True),
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_gross_exposure(jitted=dict(parallel=True)),
            pf.get_gross_exposure(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_gross_exposure(chunked=True), pf.get_gross_exposure(chunked=False))

    def test_net_exposure(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0196078431372549, -0.009900990099009901, 0.010012024441354066],
                    [0.021590645676110087, -0.02158978967815708, 0.021830620581035857],
                    [0.001519062102701967, -0.01089648232823355, 0.002949383274126105],
                    [0.0, -0.019647525359797767, 0.0],
                    [0.02467578242711193, -0.04890251030278087, 0.0],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.net_exposure, result)
        pd.testing.assert_frame_equal(pf_grouped.get_net_exposure(group_by=False), result)
        pd.testing.assert_frame_equal(
            pf_shared.get_net_exposure(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.009900990099009901, -0.004975124378109453, 0.010012024441354066],
                        [0.010896700370160913, -0.01089648232823355, 0.021830620581035857],
                        [0.0007547354365495435, -0.0054739982346853336, 0.002949383274126105],
                        [0.0, -0.00991109794697057, 0.0],
                        [0.01241841659128274, -0.024722582952177028, 0.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0049258657209004485, 0.010012024441354066],
                    [0.0, 0.021830620581035857],
                    [-0.0047572314326776634, 0.002949383274126105],
                    [-0.009993047337315064, 0.0],
                    [-0.012277657359218154, 0.0],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_net_exposure(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.net_exposure, result)
        pd.testing.assert_frame_equal(pf_shared.net_exposure, result)
        pd.testing.assert_frame_equal(
            pf.net_exposure,
            vbt.Portfolio.get_net_exposure(
                long_exposure=pf.get_gross_exposure(direction="longonly"),
                short_exposure=pf.get_gross_exposure(direction="shortonly"),
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.net_exposure,
            vbt.Portfolio.get_net_exposure(
                long_exposure=pf_grouped.get_gross_exposure(direction="longonly"),
                short_exposure=pf_grouped.get_gross_exposure(direction="shortonly"),
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.net_exposure,
            vbt.Portfolio.get_net_exposure(
                long_exposure=pf_shared.get_gross_exposure(direction="longonly"),
                short_exposure=pf_shared.get_gross_exposure(direction="shortonly"),
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_net_exposure(jitted=dict(parallel=True)),
            pf.get_net_exposure(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_net_exposure(chunked=True), pf.get_net_exposure(chunked=False))

    def test_value(self):
        result = pd.DataFrame(
            np.array(
                [
                    [102.0, 99.0, 99.8799],
                    [101.89598000000001, 97.89602, 100.77588000000002],
                    [202.83628000000004, 197.89602000000002, 101.71618000000001],
                    [202.82832000000002, 195.58798000000002, 101.70822000000001],
                    [202.62782, 194.38848000000002, 101.70822000000001],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.value, result)
        pd.testing.assert_frame_equal(pf_grouped.get_value(group_by=False), result)
        pd.testing.assert_frame_equal(
            pf_shared.get_value(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [202.0, 199.0, 99.8799],
                        [201.89597999999998, 197.89602000000002, 100.77588000000002],
                        [402.83628, 397.89602, 101.71618000000001],
                        [402.82831999999996, 395.58798, 101.70822000000001],
                        [402.62782, 394.38848, 101.70822000000001],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [201.0, 99.8799],
                    [199.792, 100.77588000000002],
                    [400.73230000000007, 101.71618000000001],
                    [398.41630000000004, 101.70822000000001],
                    [397.01630000000006, 101.70822000000001],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_value(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.value, result)
        pd.testing.assert_frame_equal(pf_shared.value, result)
        pd.testing.assert_frame_equal(
            pf.value,
            vbt.Portfolio.get_value(cash=pf.cash, asset_value=pf.asset_value, wrapper=pf.wrapper),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.value,
            vbt.Portfolio.get_value(
                cash=pf_grouped.cash,
                asset_value=pf_grouped.asset_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_value(jitted=dict(parallel=True)),
            pf.get_value(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_value(chunked=True), pf.get_value(chunked=False))

    def test_total_profit(self):
        pd.testing.assert_series_equal(pf.total_profit, (pf.value.iloc[-1] - pf.input_value).rename("total_profit"))
        pd.testing.assert_series_equal(
            pf_grouped.get_total_profit(group_by=False),
            (pf_grouped.get_value(group_by=False).iloc[-1] - pf_grouped.get_input_value(group_by=False)).rename(
                "total_profit"
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_total_profit(group_by=False),
            (pf_shared.get_value(group_by=False).iloc[-1] - pf_shared.get_input_value(group_by=False)).rename(
                "total_profit"
            ),
        )
        pd.testing.assert_series_equal(
            pf.get_total_profit(group_by=group_by),
            (pf.get_value(group_by=group_by).iloc[-1] - pf.get_input_value(group_by=group_by)).rename("total_profit"),
        )
        pd.testing.assert_series_equal(
            pf_grouped.total_profit,
            (pf_grouped.value.iloc[-1] - pf_grouped.input_value).rename("total_profit"),
        )
        pd.testing.assert_series_equal(
            pf_shared.total_profit,
            (pf_shared.value.iloc[-1] - pf_shared.input_value).rename("total_profit"),
        )
        pd.testing.assert_series_equal(
            pf.total_profit,
            vbt.Portfolio.get_total_profit(
                close=pf.filled_close,
                orders=pf.orders,
                init_position=pf.init_position,
                init_price=pf.init_price,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.total_profit,
            vbt.Portfolio.get_total_profit(
                close=pf_grouped.filled_close,
                orders=pf_grouped.orders,
                init_position=pf.init_position,
                init_price=pf.init_price,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.total_profit,
            vbt.Portfolio.get_total_profit(
                close=pf_shared.filled_close,
                orders=pf_shared.orders,
                init_position=pf.init_position,
                init_price=pf.init_price,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf.get_total_profit(jitted=dict(parallel=True)),
            pf.get_total_profit(jitted=dict(parallel=False)),
        )
        pd.testing.assert_series_equal(pf.get_total_profit(chunked=True), pf.get_total_profit(chunked=False))

    def test_final_value(self):
        pd.testing.assert_series_equal(pf.final_value, pf.value.iloc[-1].rename("final_value"))
        pd.testing.assert_series_equal(
            pf_grouped.get_final_value(group_by=False),
            pf_grouped.get_value(group_by=False).iloc[-1].rename("final_value"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_final_value(group_by=False),
            pf_shared.get_value(group_by=False).iloc[-1].rename("final_value"),
        )
        pd.testing.assert_series_equal(
            pf.get_final_value(group_by=group_by),
            pf.get_value(group_by=group_by).iloc[-1].rename("final_value"),
        )
        pd.testing.assert_series_equal(pf_grouped.final_value, pf_grouped.value.iloc[-1].rename("final_value"))
        pd.testing.assert_series_equal(pf_shared.final_value, pf_shared.value.iloc[-1].rename("final_value"))
        pd.testing.assert_series_equal(
            pf.final_value,
            vbt.Portfolio.get_final_value(input_value=pf.input_value, total_profit=pf.total_profit, wrapper=pf.wrapper),
        )
        pd.testing.assert_series_equal(
            pf_grouped.final_value,
            vbt.Portfolio.get_final_value(
                input_value=pf_grouped.input_value,
                total_profit=pf_grouped.total_profit,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.final_value,
            vbt.Portfolio.get_final_value(
                input_value=pf_shared.input_value,
                total_profit=pf_shared.total_profit,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf.get_final_value(jitted=dict(parallel=True)),
            pf.get_final_value(jitted=dict(parallel=False)),
        )
        pd.testing.assert_series_equal(pf.get_final_value(chunked=True), pf.get_final_value(chunked=False))

    def test_total_return(self):
        pd.testing.assert_series_equal(
            pf.total_return,
            ((pf.value.iloc[-1] - pf.input_value) / pf.input_value).rename("total_return"),
        )
        pd.testing.assert_series_equal(
            pf_grouped.get_total_return(group_by=False),
            (
                (pf_grouped.get_value(group_by=False).iloc[-1] - pf_grouped.get_input_value(group_by=False))
                / pf_grouped.get_input_value(group_by=False)
            ).rename("total_return"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_total_return(group_by=False),
            (
                (pf_shared.get_value(group_by=False).iloc[-1] - pf_shared.get_input_value(group_by=False))
                / pf_shared.get_input_value(group_by=False)
            ).rename("total_return"),
        )
        pd.testing.assert_series_equal(
            pf.get_total_return(group_by=group_by),
            (
                (pf.get_value(group_by=group_by).iloc[-1] - pf.get_input_value(group_by=group_by))
                / pf.get_input_value(group_by=group_by)
            ).rename("total_return"),
        )
        pd.testing.assert_series_equal(
            pf_grouped.total_return,
            ((pf_grouped.value.iloc[-1] - pf_grouped.input_value) / pf_grouped.input_value).rename("total_return"),
        )
        pd.testing.assert_series_equal(
            pf_shared.total_return,
            ((pf_shared.value.iloc[-1] - pf_shared.input_value) / pf_shared.input_value).rename("total_return"),
        )
        pd.testing.assert_series_equal(
            pf.total_return,
            vbt.Portfolio.get_total_return(
                input_value=pf.input_value,
                total_profit=pf.total_profit,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.total_return,
            vbt.Portfolio.get_total_return(
                input_value=pf_grouped.input_value,
                total_profit=pf_grouped.total_profit,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.total_return,
            vbt.Portfolio.get_total_return(
                input_value=pf_shared.input_value,
                total_profit=pf_shared.total_profit,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf.get_total_return(jitted=dict(parallel=True)),
            pf.get_total_return(jitted=dict(parallel=False)),
        )
        pd.testing.assert_series_equal(pf.get_total_return(chunked=True), pf.get_total_return(chunked=False))

    def test_returns(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, -0.0012009999999999365],
                    [-0.0010198039215685425, -0.011151313131313203, 0.008970573658964502],
                    [0.00922803824056686, 0.0, 0.009330605696521761],
                    [-3.9243472617548996e-05, -0.01166289246241539, -7.825696954011722e-05],
                    [-0.0009885207351715245, -0.006132789959792009, 0.0],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.returns, result)
        pd.testing.assert_frame_equal(pf_grouped.get_returns(group_by=False), result)
        pd.testing.assert_frame_equal(
            pf_shared.get_returns(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, -0.0012009999999999365],
                        [-0.0005149504950496028, -0.005547638190954667, 0.008970573658964502],
                        [0.0046573487991192685, 0.0, 0.009330605696521761],
                        [-1.9759888558263675e-05, -0.005800610923426691, -7.825696954011722e-05],
                        [-0.0004977306461471647, -0.003032195265386983, 0.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, -0.0012009999999999365],
                    [-0.006009950248756211, 0.008970573658964502],
                    [0.0047063946504367765, 0.009330605696521761],
                    [-0.005779419328065221, -7.825696954011722e-05],
                    [-0.003513912457898879, 0.0],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_returns(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.returns, result)
        pd.testing.assert_frame_equal(pf_shared.returns, result)
        pd.testing.assert_frame_equal(
            pf.returns,
            vbt.Portfolio.get_returns(
                init_value=pf.init_value,
                cash_deposits=pf.cash_deposits,
                value=pf.value,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.returns,
            vbt.Portfolio.get_returns(
                init_value=pf_grouped.init_value,
                cash_deposits=pf_grouped.cash_deposits,
                value=pf_grouped.value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.returns,
            vbt.Portfolio.get_returns(
                init_value=pf_shared.init_value,
                cash_deposits=pf_shared.cash_deposits,
                value=pf_shared.value,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_returns(jitted=dict(parallel=True)),
            pf.get_returns(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_returns(chunked=True), pf.get_returns(chunked=False))
        pd.testing.assert_frame_equal(
            pf_grouped.get_returns(jitted=dict(parallel=True)),
            pf_grouped.get_returns(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf_grouped.get_returns(chunked=True), pf_grouped.get_returns(chunked=False))

    def test_asset_returns(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, -0.1201000000000001],
                    [-0.05200999999999989, -1.10398, 0.8959800000000002],
                    [0.42740909090909074, 0.0, 0.42740909090909074],
                    [-0.026533333333334127, -1.0491090909090908, -0.026533333333334127],
                    [-0.04009999999999998, -0.2998749999999999, 0.0],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.asset_returns, result)
        pd.testing.assert_frame_equal(pf_grouped.get_asset_returns(group_by=False), result)
        pd.testing.assert_frame_equal(pf_shared.get_asset_returns(group_by=False), result)
        result = pd.DataFrame(
            np.array(
                [
                    [-1.0, 0.8798999999999999],
                    [-1.208, 0.8959800000000002],
                    [0.4948947368421051, 0.42740909090909074],
                    [-1.2189473684210528, -0.026533333333334127],
                    [-0.34999999999999987, 0.0],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_asset_returns(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.asset_returns, result)
        pd.testing.assert_frame_equal(pf_shared.asset_returns, result)
        pd.testing.assert_frame_equal(
            pf.asset_returns,
            vbt.Portfolio.get_asset_returns(
                init_position_value=pf.init_position_value,
                cash_flow=pf.cash_flow,
                asset_value=pf.asset_value,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.asset_returns,
            vbt.Portfolio.get_asset_returns(
                init_position_value=pf_grouped.init_position_value,
                cash_flow=pf_grouped.cash_flow,
                asset_value=pf_grouped.asset_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.asset_returns,
            vbt.Portfolio.get_asset_returns(
                init_position_value=pf_shared.init_position_value,
                cash_flow=pf_shared.cash_flow,
                asset_value=pf_shared.asset_value,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_asset_returns(jitted=dict(parallel=True)),
            pf.get_asset_returns(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(pf.get_asset_returns(chunked=True), pf.get_asset_returns(chunked=False))
        size = pd.Series([0.0, 0.5, -0.5, -0.5, 0.5, 1.0, -2.0, 2.0])
        pf2 = vbt.Portfolio.from_orders(1, size, fees=0.0)
        pd.testing.assert_series_equal(pf2.asset_returns, pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        pf3 = vbt.Portfolio.from_orders(1, size, fees=0.01)
        pd.testing.assert_series_equal(
            pf3.asset_returns,
            pd.Series(
                [
                    0.0,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.010000000000000009,
                    -0.010000000000000009,
                ]
            ),
        )

    def test_market_value(self):
        result = pd.DataFrame(
            np.array(
                [
                    [102.0, 99.0, 100.0],
                    [102.0, 198.0, 200.0],
                    [253.0, 298.0, 300.0],
                    [337.3333333333333, 596.0, 400.0],
                    [421.66666666666663, 745.0, 400.0],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.market_value, result)
        pd.testing.assert_frame_equal(pf_grouped.get_market_value(group_by=False), result)
        pd.testing.assert_frame_equal(
            pf_shared.get_market_value(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [202.0, 199.0, 100.0],
                        [202.0, 398.0, 200.0],
                        [503.0, 598.0, 300.0],
                        [670.6666666666666, 1196.0, 400.0],
                        [838.3333333333333, 1495.0, 400.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        result = pd.DataFrame(
            np.array(
                [
                    [201.0, 100.0],
                    [300.0, 200.0],
                    [551.0, 300.0],
                    [933.3333333333333, 400.0],
                    [1166.6666666666665, 400.0],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_market_value(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.market_value, result)
        pd.testing.assert_frame_equal(pf_shared.market_value, result)
        pd.testing.assert_frame_equal(
            pf.market_value,
            vbt.Portfolio.get_market_value(
                close=pf.filled_close,
                init_value=pf.get_init_value(group_by=False),
                cash_deposits=pf.get_cash_deposits(group_by=False),
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.market_value,
            vbt.Portfolio.get_market_value(
                close=pf_grouped.filled_close,
                init_value=pf_grouped.get_init_value(group_by=False),
                cash_deposits=pf_grouped.get_cash_deposits(group_by=False),
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.market_value,
            vbt.Portfolio.get_market_value(
                close=pf_shared.filled_close,
                init_value=pf_shared.get_init_value(group_by=False, split_shared=True),
                cash_deposits=pf_shared.get_cash_deposits(group_by=False, split_shared=True),
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_market_value(jitted=dict(parallel=True)),
            pf.get_market_value(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(
            pf.get_market_value(chunked=True),
            pf.get_market_value(chunked=False),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_market_value(jitted=dict(parallel=True)),
            pf_grouped.get_market_value(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.get_market_value(chunked=True),
            pf_grouped.get_market_value(chunked=False),
        )

    def test_market_returns(self):
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.5, 0.0, 0.5],
                    [0.33333333333333326, 1.0, 0.3333333333333333],
                    [0.24999999999999994, 0.25, 0.0],
                ]
            ),
            index=price_na.index,
            columns=price_na.columns,
        )
        pd.testing.assert_frame_equal(pf.market_returns, result)
        pd.testing.assert_frame_equal(pf_grouped.get_market_returns(group_by=False), result)
        pd.testing.assert_frame_equal(pf_shared.get_market_returns(group_by=False), result)
        result = pd.DataFrame(
            np.array(
                [
                    [0.0, 0.0],
                    [0.4925373134328358, 1.0],
                    [0.17, 0.5],
                    [0.6938898971566847, 0.3333333333333333],
                    [0.24999999999999994, 0.0],
                ]
            ),
            index=price_na.index,
            columns=pd.Index(["first", "second"], dtype="object", name="group"),
        )
        pd.testing.assert_frame_equal(pf.get_market_returns(group_by=group_by), result)
        pd.testing.assert_frame_equal(pf_grouped.market_returns, result)
        pd.testing.assert_frame_equal(pf_shared.market_returns, result)
        pd.testing.assert_frame_equal(
            pf.market_returns,
            vbt.Portfolio.get_market_returns(
                init_value=pf.init_value,
                cash_deposits=pf.cash_deposits,
                market_value=pf.market_value,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_grouped.market_returns,
            vbt.Portfolio.get_market_returns(
                init_value=pf_grouped.init_value,
                cash_deposits=pf_grouped.cash_deposits,
                market_value=pf_grouped.market_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.market_returns,
            vbt.Portfolio.get_market_returns(
                init_value=pf_shared.init_value,
                cash_deposits=pf_shared.cash_deposits,
                market_value=pf_shared.market_value,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.get_market_returns(jitted=dict(parallel=True)),
            pf.get_market_returns(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(
            pf.get_market_returns(chunked=True),
            pf.get_market_returns(chunked=False),
        )

    def test_bm_value(self):
        pd.testing.assert_frame_equal(
            pf.bm_value,
            pd.DataFrame(
                np.array(
                    [
                        [102.0, 99.0, 100.0],
                        [81.60000000000001, 79.2, 100.0],
                        [161.2, 179.2, 75.0],
                        [107.46666666666665, 89.6, 50.0],
                        [107.46666666666665, 44.8, 25.0],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.replace(bm_close=None).bm_value,
            pf.market_value,
        )
        assert pf.replace(bm_close=False).bm_value is None

    def test_bm_returns(self):
        pd.testing.assert_frame_equal(
            pf.bm_returns,
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [-0.19999999999999993, -0.19999999999999998, 0.0],
                        [-0.2500000000000002, 0.0, -0.25],
                        [-0.33333333333333337, -0.5, -0.3333333333333333],
                        [0.0, -0.5, -0.5],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.replace(bm_close=None).bm_returns,
            pf.market_returns,
        )
        assert pf.replace(bm_close=False).bm_returns is None

    def test_total_market_return(self):
        pd.testing.assert_series_equal(
            pf.total_market_return,
            ((pf.market_value.iloc[-1] - pf.input_value) / pf.input_value).rename("total_market_return"),
        )
        pd.testing.assert_series_equal(
            pf_grouped.get_total_market_return(group_by=False),
            (
                (pf_grouped.get_market_value(group_by=False).iloc[-1] - pf_grouped.get_input_value(group_by=False))
                / pf_grouped.get_input_value(group_by=False)
            ).rename("total_market_return"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_total_market_return(group_by=False),
            (
                (pf_shared.get_market_value(group_by=False).iloc[-1] - pf_shared.get_input_value(group_by=False))
                / pf_shared.get_input_value(group_by=False)
            ).rename("total_market_return"),
        )
        pd.testing.assert_series_equal(
            pf.get_total_market_return(group_by=group_by),
            (
                (pf.get_market_value(group_by=group_by).iloc[-1] - pf.get_input_value(group_by=group_by))
                / pf.get_input_value(group_by=group_by)
            ).rename("total_market_return"),
        )
        pd.testing.assert_series_equal(
            pf_grouped.total_market_return,
            ((pf_grouped.market_value.iloc[-1] - pf_grouped.input_value) / pf_grouped.input_value).rename(
                "total_market_return"
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.total_market_return,
            ((pf_shared.market_value.iloc[-1] - pf_shared.input_value) / pf_shared.input_value).rename(
                "total_market_return"
            ),
        )
        pd.testing.assert_series_equal(
            pf.total_market_return,
            vbt.Portfolio.get_total_market_return(
                input_value=pf.input_value,
                market_value=pf.market_value,
                wrapper=pf.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.total_market_return,
            vbt.Portfolio.get_total_market_return(
                input_value=pf_grouped.input_value,
                market_value=pf_grouped.market_value,
                wrapper=pf_grouped.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf_shared.total_market_return,
            vbt.Portfolio.get_total_market_return(
                input_value=pf_shared.input_value,
                market_value=pf_shared.market_value,
                wrapper=pf_shared.wrapper,
            ),
        )
        pd.testing.assert_series_equal(
            pf.get_total_market_return(jitted=dict(parallel=True)),
            pf.get_total_market_return(jitted=dict(parallel=False)),
        )
        pd.testing.assert_series_equal(
            pf.get_total_market_return(chunked=True),
            pf.get_total_market_return(chunked=False),
        )

    def test_return_methods(self):
        pd.testing.assert_frame_equal(pf_shared.cumulative_returns, pf_shared.cumulative_returns)
        pd.testing.assert_frame_equal(
            pf_shared.cumulative_returns,
            pd.DataFrame(
                np.array(
                    [
                        [0.0, -0.0012009999999998966],
                        [-0.0060099502487561685, 0.007758800000000177],
                        [-0.0013318407960194456, 0.017161800000000005],
                        [-0.0071035628576462395, 0.017082199999999936],
                        [-0.010592514017524146, 0.017082199999999936],
                    ]
                ),
                index=price_na.index,
                columns=pd.Index(["first", "second"], dtype="object", name="group"),
            ),
        )
        pd.testing.assert_frame_equal(
            pf_shared.get_cumulative_returns(group_by=False),
            pd.DataFrame(
                np.array(
                    [
                        [0.0, 0.0, -0.0012009999999998966],
                        [-0.0005149504950495709, -0.005547638190954718, 0.007758800000000177],
                        [0.0041400000000000325, -0.005547638190954718, 0.017161800000000005],
                        [0.004120158305503052, -0.011316069423691677, 0.017082199999999936],
                        [0.003620376930300262, -0.014313952156949417, 0.017082199999999936],
                    ]
                ),
                index=price_na.index,
                columns=price_na.columns,
            ),
        )
        pd.testing.assert_series_equal(pf_shared.sharpe_ratio, pf_shared.sharpe_ratio)
        pd.testing.assert_series_equal(
            pf_shared.sharpe_ratio,
            pd.Series(
                np.array([-8.966972200385989, 12.345065267401496]),
                index=pd.Index(["first", "second"], dtype="object", name="group"),
            ).rename("sharpe_ratio"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_sharpe_ratio(risk_free=0.01),
            pd.Series(
                np.array([-51.276434758632554, -23.91718815937344]),
                index=pd.Index(["first", "second"], dtype="object", name="group"),
            ).rename("sharpe_ratio"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_sharpe_ratio(year_freq="365D"),
            pd.Series(
                np.array([-8.966972200385989, 12.345065267401496]),
                index=pd.Index(["first", "second"], dtype="object", name="group"),
            ).rename("sharpe_ratio"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_sharpe_ratio(group_by=False),
            pd.Series(
                np.array([6.260933805237826, -19.34902167642263, 12.345065267401496]),
                index=price_na.columns,
            ).rename("sharpe_ratio"),
        )
        pd.testing.assert_series_equal(
            pf_shared.get_information_ratio(group_by=False),
            pd.Series(
                np.array([1.0384749617059628, 0.9525372423693426, 1.0199058062359245]),
                index=price_na.columns,
            ).rename("information_ratio"),
        )
        with pytest.raises(Exception):
            pf_shared.get_information_ratio(pf_shared.get_market_returns(group_by=False) * 2)
        pd.testing.assert_frame_equal(
            pf_shared.get_cumulative_returns(jitted=dict(parallel=True)),
            pf_shared.get_cumulative_returns(jitted=dict(parallel=False)),
        )
        pd.testing.assert_frame_equal(
            pf_shared.get_cumulative_returns(chunked=True),
            pf_shared.get_cumulative_returns(chunked=False),
        )

    def test_qs_methods(self):
        if qs_available:
            pd.testing.assert_series_equal(pf_shared.qs.sharpe().rename("sharpe_ratio"), pf_shared.sharpe_ratio)

    def test_stats(self):
        stats_index = pd.Index(
            [
                "Start",
                "End",
                "Period",
                "Start Value",
                "End Value",
                "Total Return [%]",
                "Benchmark Return [%]",
                "Max Gross Exposure [%]",
                "Total Fees Paid",
                "Max Drawdown [%]",
                "Max Drawdown Duration",
                "Total Trades",
                "Total Closed Trades",
                "Total Open Trades",
                "Open Trade PnL",
                "Win Rate [%]",
                "Best Trade [%]",
                "Worst Trade [%]",
                "Avg Winning Trade [%]",
                "Avg Losing Trade [%]",
                "Avg Winning Trade Duration",
                "Avg Losing Trade Duration",
                "Profit Factor",
                "Expectancy",
                "Sharpe Ratio",
                "Calmar Ratio",
                "Omega Ratio",
                "Sortino Ratio",
            ],
            dtype="object",
        )
        pd.testing.assert_series_equal(
            pf.stats(),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.0,
                        166.24150666666665,
                        -0.09944158449010732,
                        -71.66666666666667,
                        1.2135130969045893,
                        0.42916000000000004,
                        0.6276712912251518,
                        pd.Timedelta("2 days 00:00:00"),
                        2.3333333333333335,
                        1.6666666666666667,
                        0.6666666666666666,
                        -1.4678727272727272,
                        66.66666666666667,
                        -62.06261760946578,
                        -65.81967240213856,
                        91.58494359313319,
                        -374.9933222036729,
                        pd.Timedelta("3 days 00:00:00"),
                        pd.Timedelta("4 days 00:00:00"),
                        np.inf,
                        0.2866227272727273,
                        -0.25595098630477686,
                        889.6944375349927,
                        6.270976459353577,
                        49.897006624719126,
                    ]
                ),
                index=stats_index,
                name="agg_stats",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.replace(bm_close=None).stats(),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.0,
                        166.24150666666665,
                        -0.09944158449010732,
                        283.3333333333333,
                        1.2135130969045893,
                        0.42916000000000004,
                        0.6276712912251518,
                        pd.Timedelta("2 days 00:00:00"),
                        2.3333333333333335,
                        1.6666666666666667,
                        0.6666666666666666,
                        -1.4678727272727272,
                        66.66666666666667,
                        -62.06261760946578,
                        -65.81967240213856,
                        91.58494359313319,
                        -374.9933222036729,
                        pd.Timedelta("3 days 00:00:00"),
                        pd.Timedelta("4 days 00:00:00"),
                        np.inf,
                        0.2866227272727273,
                        -0.25595098630477686,
                        889.6944375349927,
                        6.270976459353577,
                        49.897006624719126,
                    ]
                ),
                index=stats_index,
                name="agg_stats",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.replace(bm_close=False).stats(),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.0,
                        166.24150666666665,
                        -0.09944158449010732,
                        1.2135130969045893,
                        0.42916000000000004,
                        0.6276712912251518,
                        pd.Timedelta("2 days 00:00:00"),
                        2.3333333333333335,
                        1.6666666666666667,
                        0.6666666666666666,
                        -1.4678727272727272,
                        66.66666666666667,
                        -62.06261760946578,
                        -65.81967240213856,
                        91.58494359313319,
                        -374.9933222036729,
                        pd.Timedelta("3 days 00:00:00"),
                        pd.Timedelta("4 days 00:00:00"),
                        np.inf,
                        0.2866227272727273,
                        -0.25595098630477686,
                        889.6944375349927,
                        6.270976459353577,
                        49.897006624719126,
                    ]
                ),
                index=pd.Index(
                    [
                        "Start",
                        "End",
                        "Period",
                        "Start Value",
                        "End Value",
                        "Total Return [%]",
                        "Max Gross Exposure [%]",
                        "Total Fees Paid",
                        "Max Drawdown [%]",
                        "Max Drawdown Duration",
                        "Total Trades",
                        "Total Closed Trades",
                        "Total Open Trades",
                        "Open Trade PnL",
                        "Win Rate [%]",
                        "Best Trade [%]",
                        "Worst Trade [%]",
                        "Avg Winning Trade [%]",
                        "Avg Losing Trade [%]",
                        "Avg Winning Trade Duration",
                        "Avg Losing Trade Duration",
                        "Profit Factor",
                        "Expectancy",
                        "Sharpe Ratio",
                        "Calmar Ratio",
                        "Omega Ratio",
                        "Sortino Ratio",
                    ],
                    dtype="object",
                ),
                name="agg_stats",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.0,
                        202.62781999999999,
                        0.3108019801980197,
                        -60.00000000000001,
                        2.467578242711193,
                        0.48618000000000006,
                        0.10277254148026709,
                        pd.Timedelta("2 days 00:00:00"),
                        3,
                        2,
                        1,
                        -0.20049999999999982,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", settings=dict(freq="10 days", year_freq="200 days")),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("50 days 00:00:00"),
                        100.0,
                        202.62781999999999,
                        0.3108019801980197,
                        -60.00000000000001,
                        2.467578242711193,
                        0.48618000000000006,
                        0.10277254148026709,
                        pd.Timedelta("20 days 00:00:00"),
                        3,
                        2,
                        1,
                        -0.20049999999999982,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("35 days 00:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        1.4651010643478568,
                        104.44254493914563,
                        4.506828421607624,
                        10.1075418640978,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", settings=dict(trade_type="positions")),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.0,
                        202.62781999999999,
                        0.3108019801980197,
                        -60.00000000000001,
                        2.467578242711193,
                        0.48618000000000006,
                        0.10277254148026709,
                        pd.Timedelta("2 days 00:00:00"),
                        3,
                        2,
                        1,
                        -0.20049999999999982,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                    ]
                ),
                index=pd.Index(
                    [
                        "Start",
                        "End",
                        "Period",
                        "Start Value",
                        "End Value",
                        "Total Return [%]",
                        "Benchmark Return [%]",
                        "Max Gross Exposure [%]",
                        "Total Fees Paid",
                        "Max Drawdown [%]",
                        "Max Drawdown Duration",
                        "Total Trades",
                        "Total Closed Trades",
                        "Total Open Trades",
                        "Open Trade PnL",
                        "Win Rate [%]",
                        "Best Trade [%]",
                        "Worst Trade [%]",
                        "Avg Winning Trade [%]",
                        "Avg Losing Trade [%]",
                        "Avg Winning Trade Duration",
                        "Avg Losing Trade Duration",
                        "Profit Factor",
                        "Expectancy",
                        "Sharpe Ratio",
                        "Calmar Ratio",
                        "Omega Ratio",
                        "Sortino Ratio",
                    ],
                    dtype="object",
                ),
                name="a",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", settings=dict(required_return=0.1, risk_free=0.01)),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.0,
                        202.62781999999999,
                        0.3108019801980197,
                        -60.00000000000001,
                        2.467578242711193,
                        0.48618000000000006,
                        0.10277254148026709,
                        pd.Timedelta("2 days 00:00:00"),
                        3,
                        2,
                        1,
                        -0.20049999999999982,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        -37.32398741973627,
                        665.2843559613844,
                        0.0,
                        -19.089875288486446,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", settings=dict(use_asset_returns=True)),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.0,
                        202.62781999999999,
                        0.3108019801980197,
                        -60.00000000000001,
                        2.467578242711193,
                        0.48618000000000006,
                        0.10277254148026709,
                        pd.Timedelta("2 days 00:00:00"),
                        3,
                        2,
                        1,
                        -0.20049999999999982,
                        100.0,
                        41.25431425976392,
                        37.25295186194369,
                        39.25363306085381,
                        np.nan,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.NaT,
                        np.inf,
                        0.4141600000000001,
                        5.746061520593739,
                        418742834.6664331,
                        3.602470352954977,
                        37.244793636996356,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", settings=dict(incl_open=True)),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        100.0,
                        202.62781999999999,
                        0.3108019801980197,
                        -60.00000000000001,
                        2.467578242711193,
                        0.48618000000000006,
                        0.10277254148026709,
                        pd.Timedelta("2 days 00:00:00"),
                        3,
                        2,
                        1,
                        -0.20049999999999982,
                        66.66666666666666,
                        41.25431425976392,
                        -3.9702970297029667,
                        39.25363306085381,
                        -3.9702970297029667,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.Timedelta("1 days 00:00:00"),
                        4.131271820448882,
                        0.20927333333333345,
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                    ]
                ),
                index=stats_index,
                name="a",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf_grouped.stats(column="first"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        200.0,
                        397.0163,
                        -0.9934413965087281,
                        -68.75458196678184,
                        0.4975124378109453,
                        0.8417000000000001,
                        0.9273023412387791,
                        pd.Timedelta("2 days 00:00:00"),
                        5,
                        3,
                        2,
                        -4.403618181818182,
                        66.66666666666666,
                        41.25431425976392,
                        -374.9933222036729,
                        39.25363306085381,
                        -374.9933222036729,
                        pd.Timedelta("3 days 12:00:00"),
                        pd.Timedelta("4 days 00:00:00"),
                        2.0281986101032405,
                        0.1399727272727273,
                        -8.966972200385989,
                        -51.016263121272715,
                        0.307541522123087,
                        -10.006463484493487,
                    ]
                ),
                index=stats_index,
                name="first",
                dtype=object,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", tags="trades and open and not closed", settings=dict(incl_open=True)),
            pd.Series(
                np.array([1, -0.20049999999999982]),
                index=pd.Index(["Total Open Trades", "Open Trade PnL"], dtype="object"),
                name="a",
                dtype=object,
            ),
        )
        max_winning_streak = (
            "max_winning_streak",
            dict(title="Max Winning Streak", calc_func=lambda trades: trades.winning_streak.max(), resolve_trades=True),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", metrics=max_winning_streak),
            pd.Series([2.0], index=["Max Winning Streak"], name="a", dtype=object),
        )
        max_winning_streak = (
            "max_winning_streak",
            dict(
                title="Max Winning Streak",
                calc_func=lambda self, group_by: self.get_trades(group_by=group_by).winning_streak.max(),
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", metrics=max_winning_streak),
            pd.Series([2.0], index=["Max Winning Streak"], name="a", dtype=object),
        )
        max_winning_streak = (
            "max_winning_streak",
            dict(
                title="Max Winning Streak",
                calc_func=lambda self, settings: self.get_trades(group_by=settings["group_by"]).winning_streak.max(),
                resolve_calc_func=False,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(column="a", metrics=max_winning_streak),
            pd.Series([2.0], index=["Max Winning Streak"], name="a", dtype=object),
        )
        vbt.settings.portfolio.stats["settings"]["my_arg"] = 100
        my_arg_metric = ("my_arg_metric", dict(title="My Arg", calc_func=lambda my_arg: my_arg))
        pd.testing.assert_series_equal(
            pf.stats(my_arg_metric, column="a"),
            pd.Series([100], index=["My Arg"], name="a", dtype=object),
        )
        vbt.settings.portfolio.stats.reset()
        pd.testing.assert_series_equal(
            pf.stats(my_arg_metric, column="a", settings=dict(my_arg=200)),
            pd.Series([200], index=["My Arg"], name="a", dtype=object),
        )
        my_arg_metric = ("my_arg_metric", dict(title="My Arg", my_arg=300, calc_func=lambda my_arg: my_arg))
        pd.testing.assert_series_equal(
            pf.stats(my_arg_metric, column="a", settings=dict(my_arg=200)),
            pd.Series([300], index=["My Arg"], name="a", dtype=object),
        )
        pd.testing.assert_series_equal(
            pf.stats(
                my_arg_metric,
                column="a",
                settings=dict(my_arg=200),
                metric_settings=dict(my_arg_metric=dict(my_arg=400)),
            ),
            pd.Series([400], index=["My Arg"], name="a", dtype=object),
        )
        trade_min_pnl_cnt = (
            "trade_min_pnl_cnt",
            dict(
                title=vbt.Sub("Trades with PnL over $$${min_pnl}"),
                calc_func=lambda trades, min_pnl: trades.apply_mask(trades.pnl.values >= min_pnl).count(),
                resolve_trades=True,
            ),
        )
        pd.testing.assert_series_equal(
            pf.stats(metrics=trade_min_pnl_cnt, column="a", metric_settings=dict(trade_min_pnl_cnt=dict(min_pnl=0))),
            pd.Series([2], index=["Trades with PnL over $0"], name="a", dtype=object),
        )
        pd.testing.assert_series_equal(
            pf.stats(
                metrics=[trade_min_pnl_cnt, trade_min_pnl_cnt, trade_min_pnl_cnt],
                column="a",
                metric_settings=dict(
                    trade_min_pnl_cnt_0=dict(min_pnl=0),
                    trade_min_pnl_cnt_1=dict(min_pnl=10),
                    trade_min_pnl_cnt_2=dict(min_pnl=20),
                ),
            ),
            pd.Series(
                [2, 0, 0],
                index=["Trades with PnL over $0", "Trades with PnL over $10", "Trades with PnL over $20"],
                name="a",
                dtype=object,
            ),
        )
        pd.testing.assert_frame_equal(
            pf.stats(metrics="total_trades", agg_func=None, settings=dict(trades_type="entry_trades")),
            pd.DataFrame([3, 2, 2], index=price_na.columns, columns=["Total Trades"]),
        )
        pd.testing.assert_frame_equal(
            pf.stats(metrics="total_trades", agg_func=None, settings=dict(trades_type="exit_trades")),
            pd.DataFrame([3, 2, 2], index=price_na.columns, columns=["Total Trades"]),
        )
        pd.testing.assert_frame_equal(
            pf.stats(metrics="total_trades", agg_func=None, settings=dict(trades_type="positions")),
            pd.DataFrame([3, 2, 2], index=price_na.columns, columns=["Total Trades"]),
        )
        pd.testing.assert_series_equal(pf["c"].stats(), pf.stats(column="c"))
        pd.testing.assert_series_equal(pf["c"].stats(), pf_grouped.stats(column="c", group_by=False))
        pd.testing.assert_series_equal(pf_grouped["second"].stats(), pf_grouped.stats(column="second"))
        pd.testing.assert_series_equal(pf_grouped["second"].stats(), pf.stats(column="second", group_by=group_by))
        pd.testing.assert_series_equal(
            pf.replace(wrapper=pf.wrapper.replace(freq="10d")).stats(),
            pf.stats(settings=dict(freq="10d")),
        )
        stats_df = pf.stats(agg_func=None)
        assert stats_df.shape == (3, 28)
        pd.testing.assert_index_equal(stats_df.index, pf.wrapper.columns)
        pd.testing.assert_index_equal(stats_df.columns, stats_index)

    def test_returns_stats(self):
        stats_index = pd.Index(
            [
                "Start",
                "End",
                "Period",
                "Total Return [%]",
                "Benchmark Return [%]",
                "Annualized Return [%]",
                "Annualized Volatility [%]",
                "Max Drawdown [%]",
                "Max Drawdown Duration",
                "Sharpe Ratio",
                "Calmar Ratio",
                "Omega Ratio",
                "Sortino Ratio",
                "Skew",
                "Kurtosis",
                "Tail Ratio",
                "Common Sense Ratio",
                "Value at Risk",
                "Alpha",
                "Beta",
            ],
            dtype="object",
        )
        pd.testing.assert_series_equal(
            pf.returns_stats(column="a"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        0.7162671975297075,
                        -60.00000000000001,
                        68.3729640692142,
                        8.374843895239454,
                        0.10277254148026353,
                        pd.Timedelta("2 days 00:00:00"),
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                        2.1657940859079745,
                        4.749360549470598,
                        7.283755486189502,
                        12.263875007651269,
                        -0.001013547284289139,
                        -0.07963807950336177,
                        -0.010617659789479695,
                    ]
                ),
                index=stats_index,
                name="a",
            ),
        )
        pd.testing.assert_series_equal(
            pf.replace(bm_close=None).returns_stats(column="a"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        0.7162671975297075,
                        150.0,
                        68.3729640692142,
                        8.374843895239454,
                        0.10277254148026353,
                        pd.Timedelta("2 days 00:00:00"),
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                        2.1657940859079745,
                        4.749360549470598,
                        7.283755486189502,
                        12.263875007651269,
                        -0.001013547284289139,
                        -0.4768429263982791,
                        0.014813148996296632,
                    ]
                ),
                index=stats_index,
                name="a",
            ),
        )
        pd.testing.assert_series_equal(
            pf.replace(bm_close=False).returns_stats(column="a"),
            pd.Series(
                np.array(
                    [
                        pd.Timestamp("2020-01-01 00:00:00"),
                        pd.Timestamp("2020-01-05 00:00:00"),
                        pd.Timedelta("5 days 00:00:00"),
                        0.7162671975297075,
                        68.3729640692142,
                        8.374843895239454,
                        0.10277254148026353,
                        pd.Timedelta("2 days 00:00:00"),
                        6.258914490528395,
                        665.2843559613844,
                        4.506828421607624,
                        43.179437771402675,
                        2.1657940859079745,
                        4.749360549470598,
                        7.283755486189502,
                        12.263875007651269,
                        -0.001013547284289139,
                    ]
                ),
                index=pd.Index(
                    [
                        "Start",
                        "End",
                        "Period",
                        "Total Return [%]",
                        "Annualized Return [%]",
                        "Annualized Volatility [%]",
                        "Max Drawdown [%]",
                        "Max Drawdown Duration",
                        "Sharpe Ratio",
                        "Calmar Ratio",
                        "Omega Ratio",
                        "Sortino Ratio",
                        "Skew",
                        "Kurtosis",
                        "Tail Ratio",
                        "Common Sense Ratio",
                        "Value at Risk",
                    ],
                    dtype="object",
                ),
                name="a",
            ),
        )

    def test_plots(self):
        pf.plot(column="a", subplots="all")
        pf.replace(bm_close=None).plot(column="a", subplots="all")
        pf.replace(bm_close=False).plot(column="a", subplots="all")
        pf.plot(column="a", subplots="all")
        pf_grouped.plot(column="first", subplots="all")
        pf_grouped.plot(column="a", subplots="all", group_by=False)
        pf_shared.plot(column="a", subplots="all", group_by=False)
        with pytest.raises(Exception):
            pf.plot(subplots="all")
        with pytest.raises(Exception):
            pf_grouped.plot(subplots="all")

    @pytest.mark.parametrize("test_freq", ["1h", "10h", "3d"])
    def test_resample(self, test_freq):
        pd.testing.assert_frame_equal(
            pf.replace(call_seq=None).resample(test_freq).returns,
            pf.returns_acc.resample(test_freq).obj,
        )
        pd.testing.assert_series_equal(
            pf.replace(call_seq=None).resample(test_freq).total_return,
            pf.total_return,
        )
        pd.testing.assert_frame_equal(
            pf_filled.replace(call_seq=None).resample(test_freq).returns,
            pf_filled.replace(call_seq=None).returns_acc.resample(test_freq).obj,
        )
        pd.testing.assert_frame_equal(
            pf_grouped.replace(call_seq=None).resample(test_freq).returns,
            pf_grouped.returns_acc.resample(test_freq).obj,
        )
        pd.testing.assert_series_equal(
            pf_grouped.replace(call_seq=None).resample(test_freq).total_return,
            pf_grouped.total_return,
        )
        pd.testing.assert_frame_equal(
            pf_grouped_filled.replace(call_seq=None).resample(test_freq).returns,
            pf_grouped_filled.replace(call_seq=None).returns_acc.resample(test_freq).obj,
        )
        pd.testing.assert_frame_equal(
            pf_shared.replace(call_seq=None).resample(test_freq).returns,
            pf_shared.returns_acc.resample(test_freq).obj,
        )
        pd.testing.assert_series_equal(
            pf_shared.replace(call_seq=None).resample(test_freq).total_return,
            pf_shared.total_return,
        )
        pd.testing.assert_frame_equal(
            pf_shared_filled.replace(call_seq=None).resample(test_freq).returns,
            pf_shared_filled.replace(call_seq=None).returns_acc.resample(test_freq).obj,
        )
        with pytest.raises(Exception):
            pf.resample(test_freq).stats()
