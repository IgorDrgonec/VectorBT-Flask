# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for working with returns.

Offers common financial risk and performance metrics as found in [empyrical](https://github.com/quantopian/empyrical),
an adapter for quantstats, and other features based on returns."""

__blacklist__ = []

try:
    import quantstats
except ImportError:
    __blacklist__.append("qs_adapter")
