import os

import pytest

import vectorbtpro as vbt
from vectorbtpro.portfolio import nb
from vectorbtpro.portfolio.call_seq import build_call_seq, build_call_seq_nb
from vectorbtpro.portfolio.enums import *
from vectorbtpro.utils.random_ import set_seed

from tests.utils import *

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
        nb.execute_order_nb(ExecState(-100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(np.nan, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, np.inf, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, np.nan, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, np.nan, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, -10.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, 0.0, np.nan, 10.0, 1100.0), nb.order_nb(10, 10))
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, size_type=-2),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, size_type=20),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, direction=-2),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, direction=20),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, -10))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, fees=np.inf))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, fees=np.nan))
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, fixed_fees=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, fixed_fees=np.nan),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, slippage=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, slippage=-1))
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, min_size=np.inf),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, min_size=-1))
    with pytest.raises(Exception):
        nb.execute_order_nb(ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0), nb.order_nb(10, 10, max_size=0))
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, max_size=-10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, size_granularity=-10),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, reject_prob=np.nan),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, reject_prob=-1),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
            nb.order_nb(10, 10, reject_prob=2),
        )

    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, np.nan),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    np.testing.assert_array_equal(
        np.asarray(account_state),
        np.asarray(ExecState(100.0, 100.0, 0.0, 100.0, 10.0, np.nan)),
    )
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=3),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, -10.0),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, -10.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=4),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, np.inf, 1100.0),
            nb.order_nb(10, 10, size_type=SizeType.Value),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, -10.0, 1100),
            nb.order_nb(10, 10, size_type=SizeType.Value),
        )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, np.nan, 1100.0),
        nb.order_nb(10, 10, size_type=SizeType.Value),
    )
    np.testing.assert_array_equal(
        np.asarray(account_state),
        np.asarray(ExecState(100.0, 100.0, 0.0, 100.0, np.nan, 1100.0)),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, np.inf, 1100.0),
            nb.order_nb(10, 10, size_type=SizeType.TargetValue),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(100.0, 100.0, 0.0, 100.0, -10.0, 1100),
            nb.order_nb(10, 10, size_type=SizeType.TargetValue),
        )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, np.nan, 1100.0),
        nb.order_nb(10, 10, size_type=SizeType.TargetValue),
    )
    np.testing.assert_array_equal(
        np.asarray(account_state),
        np.asarray(ExecState(100.0, 100.0, 0.0, 100.0, np.nan, 1100.0)),
    )
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=2),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, -10.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(np.inf, 10, direction=Direction.ShortOnly),
    )
    assert account_state == ExecState(200.0, -20.0, 100.0, 0.0, 10.0, 1100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, -10.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-np.inf, 10, direction=Direction.Both),
    )
    assert account_state == ExecState(200.0, -20.0, 100.0, 0.0, 10.0, 1100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 10.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(0, 10),
    )
    assert account_state == ExecState(100.0, 10.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=5),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(15, 10, max_size=10, allow_partial=False),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=9),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(10, 10, reject_prob=1.0),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=10),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(0.0, 100.0, 0.0, 0.0, 10.0, 1100.0),
        nb.order_nb(10, 10, direction=Direction.LongOnly),
    )
    assert account_state == ExecState(0.0, 100.0, 0.0, 0.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=7),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(0.0, 100.0, 0.0, 0.0, 10.0, 1100.0),
        nb.order_nb(10, 10, direction=Direction.Both),
    )
    assert account_state == ExecState(0.0, 100.0, 0.0, 0.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=7),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(np.inf, 100, 0.0, np.inf, np.nan, 1100.0),
            nb.order_nb(np.inf, 10, direction=Direction.LongOnly),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(np.inf, 100.0, 0.0, np.inf, 10.0, 1100.0),
            nb.order_nb(np.inf, 10, direction=Direction.Both),
        )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-10, 10, direction=Direction.ShortOnly),
    )
    assert account_state == ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=8),
    )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(np.inf, 100.0, 0.0, np.inf, 10.0, 1100.0),
            nb.order_nb(-np.inf, 10, direction=Direction.ShortOnly),
        )
    with pytest.raises(Exception):
        nb.execute_order_nb(
            ExecState(np.inf, 100.0, 0.0, np.inf, 10.0, 1100.0),
            nb.order_nb(-np.inf, 10, direction=Direction.Both),
        )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-10, 10, direction=Direction.LongOnly),
    )
    assert account_state == ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=8),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(10, 10, fixed_fees=100),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=11),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(10, 10, min_size=100),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=12),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(100, 10, allow_partial=False),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=13),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-10, 10, min_size=100),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=12),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-200, 10, direction=Direction.LongOnly, allow_partial=False),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=13),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0),
        nb.order_nb(-10, 10, fixed_fees=1000),
    )
    assert account_state == ExecState(100.0, 100.0, 0.0, 100.0, 10.0, 1100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=11),
    )

    # Calculations
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(10, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(0.0, 8.18181818181818, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=8.18181818181818, price=11.0, fees=10.000000000000014, side=0, status=0, status_info=-1),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(100, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(0.0, 8.18181818181818, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=8.18181818181818, price=11.0, fees=10.000000000000014, side=0, status=0, status_info=-1),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-10, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(180.0, -10.0, 90.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=9.0, fees=10.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-100, 10, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(909.0, -100.0, 900.0, -891.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=100.0, price=9.0, fees=91.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(10, 10, fees=-0.1, fixed_fees=-1, slippage=0.1),
    )
    assert account_state == ExecState(2.0, 10.0, 0.0, 2.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=11.0, fees=-12.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(10, 0, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(99.0, 10.0, 0.0, 99.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=0.0, fees=1.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 10.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-10, 0, fees=0.1, fixed_fees=1, slippage=0.1),
    )
    assert account_state == ExecState(99.0, 0.0, 0.0, 99.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=0.0, fees=1.0, side=1, status=0, status_info=-1))

    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(10, 10, size_type=SizeType.TargetAmount),
    )
    assert account_state == ExecState(0.0, 10.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-10, 10, size_type=SizeType.TargetAmount),
    )
    assert account_state == ExecState(200.0, -10.0, 100.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(100, 10, size_type=SizeType.Value),
    )
    assert account_state == ExecState(0.0, 10.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-100, 10, size_type=SizeType.Value),
    )
    assert account_state == ExecState(200.0, -10.0, 100.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(100, 10, size_type=SizeType.TargetValue),
    )
    assert account_state == ExecState(0.0, 10.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-100, 10, size_type=SizeType.TargetValue),
    )
    assert account_state == ExecState(200.0, -10.0, 100.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(1, 10, size_type=SizeType.TargetPercent),
    )
    assert account_state == ExecState(0.0, 10.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-1, 10, size_type=SizeType.TargetPercent),
    )
    assert account_state == ExecState(200.0, -10.0, 100.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, 5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(0.0, 10.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, 5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(25.0, 7.5, 0.0, 25.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, 5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(125.0, -2.5, 25.0, 75.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=7.5, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, 5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(200.0, -10.0, 100.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=15.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(0.0, 5.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(25.0, 2.5, 0.0, 25.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(75.0, -2.5, 25.0, 25.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(100.0, -5.0, 50.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, -5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(0.0, 0.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, -5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(25.0, -2.5, 0.0, 25.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, -5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-0.5, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(75.0, -7.5, 25.0, 25.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=2.5, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(50.0, -5.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(-1, 10, size_type=SizeType.Percent),
    )
    assert account_state == ExecState(100.0, -10.0, 50.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(np.inf, 10),
    )
    assert account_state == ExecState(0.0, 10.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, -5.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(np.inf, 10),
    )
    assert account_state == ExecState(0.0, 5.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-np.inf, 10),
    )
    assert account_state == ExecState(200.0, -10.0, 100.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=10.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(150.0, -5.0, 0.0, 150.0, 10.0, 100.0),
        nb.order_nb(-np.inf, 10),
    )
    assert account_state == ExecState(300.0, -20.0, 150.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=15.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))

    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, 0.0, 0.0, 50.0, 10.0, 100.0),
        nb.order_nb(10, 10, lock_cash=True),
    )
    assert account_state == ExecState(50.0, 5.0, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=5.0, price=10.0, fees=0.0, side=0, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(1000.0, -5.0, 50.0, 50.0, 10.0, 100.0),
        nb.order_nb(10, 17.5, lock_cash=True),
    )
    assert account_state == ExecState(850.0, 3.571428571428571, 0.0, 0.0, 10.0, 100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=8.571428571428571, price=17.5, fees=0.0, side=0, status=0, status_info=-1),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(100.0, -5.0, 50.0, 50.0, 10.0, 100.0),
        nb.order_nb(10, 100, lock_cash=True),
    )
    assert account_state == ExecState(37.5, -4.375, 43.75, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=0.625, price=100.0, fees=0.0, side=0, status=0, status_info=-1))

    account_state, order_result = nb.execute_order_nb(
        ExecState(0.0, 10.0, 0.0, -50.0, 10.0, 100.0),
        nb.order_nb(-20, 10, lock_cash=True),
    )
    assert account_state == ExecState(150.0, -5.0, 50.0, 0.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=15.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(0.0, 1.0, 0.0, -50.0, 10.0, 100.0),
        nb.order_nb(-10, 10, lock_cash=True),
    )
    assert account_state == ExecState(10.0, 0.0, 0.0, -40.0, 10.0, 100.0)
    assert_same_tuple(order_result, OrderResult(size=1.0, price=10.0, fees=0.0, side=1, status=0, status_info=-1))
    account_state, order_result = nb.execute_order_nb(
        ExecState(0.0, 0.0, 0.0, -100.0, 10.0, 100.0),
        nb.order_nb(-10, 10, lock_cash=True),
    )
    assert account_state == ExecState(0.0, 0.0, 0.0, -100.0, 10.0, 100.0)
    assert_same_tuple(
        order_result,
        OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=6),
    )
    account_state, order_result = nb.execute_order_nb(
        ExecState(0.0, 0.0, 0.0, 100.0, 10.0, 100.0),
        nb.order_nb(-20, 10, fees=0.1, slippage=0.1, fixed_fees=1.0, lock_cash=True),
    )
    assert account_state == ExecState(80.0, -10.0, 90.0, 0.0, 10.0, 100.0)
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
