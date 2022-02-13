# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for signals.

Defines enums and other schemas for `vectorbtpro.signals`."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__all__ = ["StopType", "FactoryMode"]

__pdoc__ = {}


# ############# Enums ############# #


class StopTypeT(tp.NamedTuple):
    StopLoss: int = 0
    TrailStop: int = 1
    TakeProfit: int = 2


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
