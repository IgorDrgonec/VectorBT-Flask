# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for signals.

Defines enums and other schemas for `vectorbtpro.signals`."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__all__ = ["StopType", "FactoryMode", "GenEnContext", "GenExContext", "GenEnExContext"]

__pdoc__ = {}


# ############# Enums ############# #


class StopTypeT(tp.NamedTuple):
    SL: int = 0
    TSL: int = 1
    TP: int = 2
    TTP: int = 3


StopType = StopTypeT()
"""_"""

__pdoc__[
    "StopType"
] = f"""Stop type.

```python
{prettify(StopType)}
```
"""


class FactoryModeT(tp.NamedTuple):
    Entries: int = 0
    Exits: int = 1
    Both: int = 2
    Chain: int = 3


FactoryMode = FactoryModeT()
"""_"""

__pdoc__[
    "FactoryMode"
] = f"""Factory mode.

```python
{prettify(FactoryMode)}
```

Attributes:
    Entries: Generate entries only using `generate_func`.
    
        Takes no input signal arrays.
        Produces one output signal array - `entries`.
        
        Such generators often have no suffix.
    Exits: Generate exits only using `generate_ex_func`.
        
        Takes one input signal array - `entries`.
        Produces one output signal array - `exits`.
        
        Such generators often have suffix 'X'.
    Both: Generate both entries and exits using `generate_enex_func`.
            
        Takes no input signal arrays.
        Produces two output signal arrays - `entries` and `exits`.
        
        Such generators often have suffix 'NX'.
    Chain: Generate chain of entries and exits using `generate_enex_func`.
                
        Takes one input signal array - `entries`.
        Produces two output signal arrays - `new_entries` and `exits`.
        
        Such generators often have suffix 'CX'.
"""

# ############# Named tuples ############# #


class GenEnContext(tp.NamedTuple):
    target_shape: tp.Shape
    entries_out: tp.Array2d
    out: tp.Array1d
    from_i: int
    to_i: int
    col: int


__pdoc__["GenEnContext"] = "Context of an entry signal generator."
__pdoc__["GenEnContext.target_shape"] = "Target shape."
__pdoc__["GenEnContext.entries_out"] = "Output array with entries."
__pdoc__["GenEnContext.out"] = "Current segment of the output array with entries."
__pdoc__["GenEnContext.from_i"] = "Start index of the segment (inclusive)."
__pdoc__["GenEnContext.to_i"] = "End index of the segment (exclusive)."
__pdoc__["GenEnContext.to_i"] = "Column of the segment."


class GenExContext(tp.NamedTuple):
    entries: tp.Array2d
    until_next: bool
    skip_until_exit: bool
    exits_out: tp.Array2d
    out: tp.Array1d
    wait: int
    from_i: int
    to_i: int
    col: int


__pdoc__["GenExContext"] = "Context of an exit signal generator."
__pdoc__["GenExContext.entries"] = "Input array with entries."
__pdoc__["GenExContext.until_next"] = "Whether to place signals up to the next entry signal."
__pdoc__["GenExContext.skip_until_exit"] = "Whether to skip processing entry signals until the next exit."
__pdoc__["GenExContext.exits_out"] = "Output array with exits."
__pdoc__["GenExContext.out"] = "Current segment of the output array with exits."
__pdoc__["GenExContext.wait"] = "Number of ticks to wait before placing exits."
__pdoc__["GenExContext.from_i"] = "Start index of the segment (inclusive)."
__pdoc__["GenExContext.to_i"] = "End index of the segment (exclusive)."
__pdoc__["GenExContext.col"] = "Column of the segment."


class GenEnExContext(tp.NamedTuple):
    target_shape: tp.Shape
    entry_wait: int
    exit_wait: int
    entries_out: tp.Array2d
    exits_out: tp.Array2d
    entries_turn: bool
    wait: int
    out: tp.Array1d
    from_i: int
    to_i: int
    col: int


__pdoc__["GenExContext"] = "Context of an entry/exit signal generator."
__pdoc__["GenExContext.target_shape"] = "Target shape."
__pdoc__["GenExContext.entry_wait"] = "Number of ticks to wait before placing entries."
__pdoc__["GenExContext.exit_wait"] = "Number of ticks to wait before placing exits."
__pdoc__["GenExContext.entries_out"] = "Output array with entries."
__pdoc__["GenExContext.exits_out"] = "Output array with exits."
__pdoc__["GenExContext.entries_turn"] = "Whether the current turn is generating an entry."
__pdoc__["GenExContext.out"] = "Current segment of the output array with entries/exits."
__pdoc__["GenExContext.wait"] = "Number of ticks to wait before placing entries/exits."
__pdoc__["GenExContext.from_i"] = "Start index of the segment (inclusive)."
__pdoc__["GenExContext.to_i"] = "End index of the segment (exclusive)."
__pdoc__["GenExContext.col"] = "Column of the segment."



