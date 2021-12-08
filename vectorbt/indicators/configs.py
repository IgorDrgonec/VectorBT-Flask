# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Configs."""

from vectorbt.utils.config import ReadonlyConfig

flex_elem_param_config = ReadonlyConfig(
    dict(
        is_array_like=True,  # passing a NumPy array means passing one value, for multiple use list
        bc_to_input=True,  # broadcast to input
        broadcast_kwargs=dict(
            keep_flex=True  # keep original shape for flexible indexing to save memory
        )
    )
)
"""Config for flexible element-wise parameters."""

flex_col_param_config = ReadonlyConfig(
    dict(
        is_array_like=True,
        bc_to_input=1,  # broadcast to axis 1 (columns)
        per_column=True,  # display one parameter per column
        broadcast_kwargs=dict(
            keep_flex=True
        )
    )
)
"""Config for flexible column-wise parameters."""
