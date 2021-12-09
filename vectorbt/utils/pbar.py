# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Utilities for showing progress bar."""

from vectorbt import _typing as tp
from vectorbt.utils.config import merge_dicts


def get_pbar(*args, pbar_type: tp.Optional[str] = None, show_progress: bool = True, **kwargs):
    """Get a `tqdm` progress bar.

    Supported types:

    * 'tqdm_auto'
    * 'tqdm_notebook'
    * 'tqdm_gui'
    * 'tqdm'

    For defaults, see `pbar` in `vectorbt._settings.settings`."""

    from vectorbt._settings import settings
    pbar_cfg = settings['pbar']

    if pbar_cfg['disable']:
        show_progress = False
    if pbar_type is None:
        pbar_type = pbar_cfg['type']
    kwargs = merge_dicts(pbar_cfg['kwargs'], kwargs)

    if pbar_type.lower() == 'tqdm_auto':
        from tqdm.auto import tqdm as pbar
    elif pbar_type.lower() == 'tqdm_notebook':
        from tqdm.notebook import tqdm as pbar
    elif pbar_type.lower() == 'tqdm_gui':
        from tqdm.gui import tqdm as pbar
    elif pbar_type.lower() == 'tqdm':
        from tqdm import tqdm as pbar
    else:
        raise ValueError(f"pbar_type cannot be '{pbar_type}'")
    return pbar(*args, disable=not show_progress, **kwargs)
