import os

import numpy as np
import pytest

import vectorbtpro as vbt
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.call_seq import build_call_seq, build_call_seq_nb
from vectorbtpro.portfolio.enums import *
from vectorbtpro.utils.random_ import set_seed

seed = 42


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
