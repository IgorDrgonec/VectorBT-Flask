# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Global settings.

`settings` config is also accessible via `vectorbt.settings`.

!!! note
    All places in vectorbt import `settings` from `vectorbt._settings.settings`, not from `vectorbt`.
    Overwriting `vectorbt.settings` only overwrites the reference created for the user.
    Consider updating the settings config instead of replacing it.

Here are the main properties of the `settings` config:

* It's a nested config, that is, a config that consists of multiple sub-configs.
    one per sub-package (e.g., 'data'), module (e.g., 'wrapping'), or even class (e.g., 'configured').
    Each sub-config may consist of other sub-configs.
* It has frozen keys - you cannot add other sub-configs or remove the existing ones, but you can modify them.
* Each sub-config can either inherit the properties of the parent one by using `dict` or overwrite them
    by using its own `vectorbt.utils.config.Config`. The main reason for defining an own config is to allow
    adding new keys (e.g., 'plotting.layout').

For example, you can change default width and height of each plot:

```python-repl
>>> import vectorbt as vbt

>>> vbt.settings['plotting']['layout']['width'] = 800
>>> vbt.settings['plotting']['layout']['height'] = 400
```

The main sub-configs such as for plotting can be also accessed/modified using the dot notation:

```
>>> vbt.settings.plotting['layout']['width'] = 800
```

Some sub-configs allow the dot notation too but this depends whether they inherit the rules of the root config.

```plaintext
>>> vbt.settings.data - ok
>>> vbt.settings.data.binance - ok
>>> vbt.settings.data.binance.api_key - error
>>> vbt.settings.data.binance['api_key'] - ok
```

Since this is only visible when looking at the source code, the advice is to always use the bracket notation.

!!! note
    Whether the change takes effect immediately depends upon the place that accesses the settings.
    For example, changing 'wrapping.freq` has an immediate effect because the value is resolved
    every time `vectorbt.base.wrapping.ArrayWrapper.freq` is called. On the other hand, changing
    'portfolio.fillna_close' has only effect on `vectorbt.portfolio.base.Portfolio` instances created
    in the future, not the existing ones, because the value is resolved upon the object's construction.
    Last but not least, some settings are only accessed when importing the package for the first time,
    such as 'jitting.jit_decorator'. In any case, make sure to check whether the update actually took place.

## Saving and loading

Like any other class subclassing `vectorbt.utils.config.Config`, we can save settings to the disk,
load it back, and replace in-place:

```python-repl
>>> vbt.settings.save('my_settings')
>>> vbt.settings['caching']['disable'] = True
>>> vbt.settings['caching']['disable']
True

>>> vbt.settings.load_update('my_settings', clear=True)  # replace in-place
>>> vbt.settings['caching']['disable']
False
```

Bonus: You can do the same with any sub-config inside `settings`!

Some settings (such as Numba-related ones) are applied on import, so changing them during the runtime
will have no effect. In this case, change the settings, save them to the disk, and create an environment
variable that holds the path to the file - vectorbt will load it before any other module.

The following environment variables are supported:

* "VBT_SETTINGS_PATH": Path to the settings file. Will replace the current settings.
* "VBT_SETTINGS_OVERRIDE_PATH": Path to the settings file. Will override the current settings.

!!! note
    The environment variable must be set before importing vectorbt.
"""

import json
import os
import pkgutil

import numpy as np

from vectorbt.utils.config import Config
from vectorbt.utils.datetime_ import get_local_tz, get_utc_tz
from vectorbt.utils.execution import SequenceEngine, DaskEngine, RayEngine
from vectorbt.utils.jitting import NumPyJitter, NumbaJitter
from vectorbt.utils.template import Sub, RepEval, deep_substitute

__pdoc__ = {}

# ############# Settings sub-configs ############# #

_settings = {}

caching = dict(
    disable=False,
    disable_whitelist=False,
    disable_machinery=False,
    silence_warnings=False,
    register_lazily=True,
    ignore_args=[
        'jitted',
        'chunked'
    ],
    use_cached_accessors=True
)
"""_"""

__pdoc__['caching'] = Sub("""Sub-config with settings applied across `vectorbt.ca_registry`, 
`vectorbt.utils.caching`, and cacheable decorators in `vectorbt.utils.decorators`.

!!! hint
    Apply setting `register_lazily` on startup to register all unbound cacheables.
    
    Setting `use_cached_accessors` is applied only once.

```json
${config_doc}
```""")

_settings['caching'] = caching

jitting = dict(
    disable=False,
    disable_wrapping=False,
    disable_resolution=False,
    option=True,
    allow_new=False,
    register_new=False,
    jitters=Config(  # flex
        dict(
            nb=dict(
                cls=NumbaJitter,
                aliases={'numba'},
                options=dict(),
                override_options=dict(),
                resolve_kwargs=dict(),
                tasks=dict()
            ),
            np=dict(
                cls=NumPyJitter,
                aliases={'numpy'},
                options=dict(),
                override_options=dict(),
                resolve_kwargs=dict(),
                tasks=dict()
            )
        )
    ),
    template_mapping=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['jitting'] = Sub("""Sub-config with settings applied across `vectorbt.jit_registry` and 
`vectorbt.utils.jitting`.

!!! note
    Options (with `_options` suffix) are applied only once. 
    
    Keyword arguments (with `_kwargs` suffix) are applied right away.

```json
${config_doc}
```""")

_settings['jitting'] = jitting

numba = dict(
    parallel=None,
    silence_warnings=False,
    check_func_type=True,
    check_func_suffix=False
)
"""_"""

__pdoc__['numba'] = Sub("""Sub-config with Numba-related settings.

```json
${config_doc}
```""")

_settings['numba'] = numba

math = dict(
    use_tol=True,
    rel_tol=1e-9,  # 1,000,000,000 == 1,000,000,001
    abs_tol=1e-12  # 0.000000000001 == 0.000000000002
)
"""_"""

__pdoc__['math'] = Sub("""Sub-config with settings applied across `vectorbt.utils.math_`.

!!! note
    All math settings are applied only once.

```json
${config_doc}
```""")

_settings['math'] = math

execution = dict(
    engines=Config(  # flex
        dict(
            sequence=dict(
                cls=SequenceEngine,
                show_progress=False,
                pbar_kwargs=Config(  # flex
                    dict()
                )
            ),
            dask=dict(
                cls=DaskEngine,
                compute_kwargs=Config(  # flex
                    dict()
                ),
            ),
            ray=dict(
                cls=RayEngine,
                restart=False,
                reuse_refs=True,
                del_refs=True,
                shutdown=False,
                init_kwargs=Config(  # flex
                    dict()
                ),
                remote_kwargs=Config(  # flex
                    dict()
                )
            ),
        )
    ),
)
"""_"""

__pdoc__['execution'] = Sub("""Sub-config with settings applied across `vectorbt.utils.execution`.

```json
${config_doc}
```""")

_settings['execution'] = execution

chunking = dict(
    disable=False,
    disable_wrapping=False,
    option=False,
    engine='sequence',
    n_chunks=None,
    min_size=None,
    chunk_len=None,
    skip_one_chunk=True,
    silence_warnings=False,
    template_mapping=Config(  # flex
        dict()
    ),
    options=Config(  # flex
        dict()
    ),
    override_setup_options=Config(  # flex
        dict()
    ),
    override_options=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['chunking'] = Sub("""Sub-config with settings applied across `vectorbt.ch_registry` 
and `vectorbt.utils.chunking`.

!!! note
    Options (with `_options` suffix) and setting `disable_machinery` are applied only once.

```json
${config_doc}
```""")

_settings['chunking'] = chunking

template = dict(
    strict=True,
    mapping=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['template'] = Sub("""Sub-config with settings applied across `vectorbt.utils.template`.

```json
${config_doc}
```""")

_settings['template'] = template

config = Config(  # flex
    dict()
)
"""_"""

__pdoc__['config'] = Sub("""Sub-config with settings applied to `vectorbt.utils.config.Config`.

```json
${config_doc}
```""")

_settings['config'] = config

configured = dict(
    config=Config(  # flex
        dict(
            readonly_=True
        )
    ),
)
"""_"""

__pdoc__['configured'] = Sub("""Sub-config with settings applied to `vectorbt.utils.config.Configured`.

```json
${config_doc}
```""")

_settings['configured'] = configured

broadcasting = dict(
    align_index=False,
    align_columns=True,
    index_from='strict',
    columns_from='stack',
    ignore_sr_names=True,
    drop_duplicates=True,
    keep='last',
    drop_redundant=True,
    ignore_ranges=True,
    keep_wrap_default=False,
    keep_flex=False,
    min_one_dim=True,
    index_to_product=True,
    keys_from_sr_index=True
)
"""_"""

__pdoc__['broadcasting'] = Sub("""Sub-config with settings applied to broadcasting functions across `vectorbt.base`.

```json
${config_doc}
```""")

_settings['broadcasting'] = broadcasting

wrapping = dict(
    column_only_select=False,
    group_select=True,
    freq=None,
    silence_warnings=False
)
"""_"""

__pdoc__['wrapping'] = Sub("""Sub-config with settings applied across `vectorbt.base.wrapping`.

```json
${config_doc}
```""")

_settings['wrapping'] = wrapping

datetime = dict(
    naive_tz=get_local_tz(),
    to_py_timezone=True
)
"""_"""

__pdoc__['datetime'] = Sub("""Sub-config with settings applied across `vectorbt.utils.datetime_`.

```json
${config_doc}
```""")

_settings['datetime'] = datetime

data = dict(
    show_progress=True,
    pbar_kwargs=Config(  # flex
        dict()
    ),
    tz_localize=get_utc_tz(),
    tz_convert=get_utc_tz(),
    missing_index='nan',
    missing_columns='raise',
    custom=Config(  # flex
        dict(
            binance=dict(
                dict(
                    api_key=None,
                    api_secret=None
                )
            ),
            ccxt=dict(
                dict(
                    enableRateLimit=True
                )
            )
        )
    ),
    stats=Config(  # flex
        dict()
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['data'] = Sub("""Sub-config with settings applied across `vectorbt.data`.

```json
${config_doc}
```

## Binance

See `binance.client.Client`.

## CCXT

See [Configuring API Keys](https://ccxt.readthedocs.io/en/latest/manual.html#configuring-api-keys).
Keys can be defined per exchange. If a key is defined at the root, it applies to all exchanges.""")

_settings['data'] = data

plotting = dict(
    use_widgets=True,
    show_kwargs=Config(  # flex
        dict()
    ),
    use_gl=None,
    color_schema=Config(  # flex
        dict(
            increasing="#1b9e76",
            decreasing="#d95f02"
        )
    ),
    contrast_color_schema=Config(  # flex
        dict(
            blue="#4285F4",
            orange="#FFAA00",
            green="#37B13F",
            red="#EA4335",
            gray="#E2E2E2"
        )
    ),
    themes=dict(
        light=dict(
            color_schema=Config(  # flex
                dict(
                    blue="#1f77b4",
                    orange="#ff7f0e",
                    green="#2ca02c",
                    red="#dc3912",
                    purple="#9467bd",
                    brown="#8c564b",
                    pink="#e377c2",
                    gray="#7f7f7f",
                    yellow="#bcbd22",
                    cyan="#17becf"
                )
            ),
            template=None
        ),
        dark=dict(
            color_schema=Config(  # flex
                dict(
                    blue="#1f77b4",
                    orange="#ff7f0e",
                    green="#2ca02c",
                    red="#dc3912",
                    purple="#9467bd",
                    brown="#8c564b",
                    pink="#e377c2",
                    gray="#7f7f7f",
                    yellow="#bcbd22",
                    cyan="#17becf"
                )
            ),
            template=None
        ),
        seaborn=dict(
            color_schema=Config(  # flex
                dict(
                    blue="rgb(76,114,176)",
                    orange="rgb(221,132,82)",
                    green="rgb(129,114,179)",
                    red="rgb(85,168,104)",
                    purple="rgb(218,139,195)",
                    brown="rgb(204,185,116)",
                    pink="rgb(140,140,140)",
                    gray="rgb(100,181,205)",
                    yellow="rgb(147,120,96)",
                    cyan="rgb(196,78,82)"
                )
            ),
            template=None
        ),
    ),
    layout=Config(  # flex
        dict(
            width=700,
            height=350,
            margin=dict(
                t=30, b=30, l=30, r=30
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                traceorder='normal'
            )
        )
    ),
)
"""_"""

__pdoc__['plotting'] = Sub("""Sub-config with settings applied to Plotly figures 
created from `vectorbt.utils.figure`.

```json
${config_doc}
```
""")

_settings['plotting'] = plotting

stats_builder = dict(
    metrics='all',
    tags='all',
    silence_warnings=False,
    template_mapping=Config(  # flex
        dict()
    ),
    filters=Config(  # flex
        dict(
            is_not_grouped=dict(
                filter_func=lambda self, metric_settings:
                not self.wrapper.grouper.is_grouped(group_by=metric_settings['group_by']),
                warning_message=Sub("Metric '$metric_name' does not support grouped data")
            ),
            has_freq=dict(
                filter_func=lambda self, metric_settings:
                self.wrapper.freq is not None,
                warning_message=Sub("Metric '$metric_name' requires frequency to be set")
            )
        )
    ),
    settings=Config(  # flex
        dict(
            to_timedelta=None,
            use_caching=True
        )
    ),
    metric_settings=Config(  # flex
        dict()
    ),
)
"""_"""

__pdoc__['stats_builder'] = """Sub-config with settings applied to 
`vectorbt.generic.stats_builder.StatsBuilderMixin`.

```json
${config_doc}
```"""

_settings['stats_builder'] = stats_builder

plots_builder = dict(
    subplots='all',
    tags='all',
    silence_warnings=False,
    template_mapping=Config(  # flex
        dict()
    ),
    filters=Config(  # flex
        dict(
            is_not_grouped=dict(
                filter_func=lambda self, subplot_settings:
                not self.wrapper.grouper.is_grouped(group_by=subplot_settings['group_by']),
                warning_message=Sub("Subplot '$subplot_name' does not support grouped data")
            ),
            has_freq=dict(
                filter_func=lambda self, subplot_settings:
                self.wrapper.freq is not None,
                warning_message=Sub("Subplot '$subplot_name' requires frequency to be set")
            )
        )
    ),
    settings=Config(  # flex
        dict(
            use_caching=True,
            hline_shape_kwargs=dict(
                type='line',
                line=dict(
                    color='gray',
                    dash="dash",
                )
            )
        )
    ),
    subplot_settings=Config(  # flex
        dict()
    ),
    show_titles=True,
    hide_id_labels=True,
    group_id_labels=True,
    make_subplots_kwargs=Config(  # flex
        dict()
    ),
    layout_kwargs=Config(  # flex
        dict()
    ),
)
"""_"""

__pdoc__['plots_builder'] = Sub("""Sub-config with settings applied to 
`vectorbt.generic.plots_builder.PlotsBuilderMixin`.

```json
${config_doc}
```""")

_settings['plots_builder'] = plots_builder

generic = dict(
    use_jitted=False,
    stats=Config(  # flex
        dict(
            filters=dict(
                has_mapping=dict(
                    filter_func=lambda self, metric_settings:
                    metric_settings.get('mapping', self.mapping) is not None
                )
            ),
            settings=dict(
                incl_all_keys=False
            )
        )
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['generic'] = Sub("""Sub-config with settings applied to `vectorbt.generic.accessors.GenericAccessor`.

```json
${config_doc}
```""")

_settings['generic'] = generic

ranges = dict(
    stats=Config(  # flex
        dict()
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['ranges'] = Sub("""Sub-config with settings applied to `vectorbt.generic.ranges.Ranges`.

```json
${config_doc}
```""")

_settings['ranges'] = ranges

drawdowns = dict(
    stats=Config(  # flex
        dict(
            settings=dict(
                incl_active=False
            )
        )
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['drawdowns'] = Sub("""Sub-config with settings applied to `vectorbt.generic.drawdowns.Drawdowns`.

```json
${config_doc}
```""")

_settings['drawdowns'] = drawdowns

ohlcv = dict(
    plot_type='OHLC',
    column_names=dict(
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume'
    ),
    stats=Config(  # flex
        dict()
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['ohlcv'] = Sub("""Sub-config with settings applied to `vectorbt.ohlcv_accessors.OHLCVDFAccessor`.

```json
${config_doc}
```""")

_settings['ohlcv'] = ohlcv

signals = dict(
    stats=Config(
        dict(
            filters=dict(
                silent_has_other=dict(
                    filter_func=lambda self, metric_settings:
                    metric_settings.get('other', None) is not None
                ),
            ),
            settings=dict(
                other=None,
                other_name='Other',
                from_other=False
            )
        )
    ),  # flex
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['signals'] = Sub("""Sub-config with settings applied to `vectorbt.signals.accessors.SignalsAccessor`.

```json
${config_doc}
```""")

_settings['signals'] = signals

returns = dict(
    year_freq='365 days',
    defaults=Config(  # flex
        dict(
            start_value=0.,
            window=10,
            minp=None,
            ddof=1,
            risk_free=0.,
            levy_alpha=2.,
            required_return=0.,
            cutoff=0.05
        )
    ),
    stats=Config(  # flex
        dict(
            filters=dict(
                has_year_freq=dict(
                    filter_func=lambda self, metric_settings:
                    self.year_freq is not None,
                    warning_message=Sub("Metric '$metric_name' requires year frequency to be set")
                ),
                has_benchmark_rets=dict(
                    filter_func=lambda self, metric_settings:
                    metric_settings.get('benchmark_rets', self.benchmark_rets) is not None,
                    warning_message=Sub("Metric '$metric_name' requires benchmark_rets to be set")
                )
            ),
            settings=dict(
                check_is_not_grouped=True
            )
        )
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['returns'] = Sub("""Sub-config with settings applied to `vectorbt.returns.accessors.ReturnsAccessor`.

```json
${config_doc}
```""")

_settings['returns'] = returns

qs_adapter = dict(
    defaults=Config(  # flex
        dict()
    ),
)
"""_"""

__pdoc__['qs_adapter'] = Sub("""Sub-config with settings applied to `vectorbt.returns.qs_adapter.QSAdapter`.

```json
${config_doc}
```""")

_settings['qs_adapter'] = qs_adapter

records = dict(
    stats=Config(  # flex
        dict()
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['records'] = Sub("""Sub-config with settings applied to `vectorbt.records.base.Records`.

```json
${config_doc}
```""")

_settings['records'] = records

mapped_array = dict(
    stats=Config(  # flex
        dict(
            filters=dict(
                has_mapping=dict(
                    filter_func=lambda self, metric_settings:
                    metric_settings.get('mapping', self.mapping) is not None
                )
            ),
            settings=dict(
                incl_all_keys=False
            )
        )
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['mapped_array'] = Sub("""Sub-config with settings applied to `vectorbt.records.mapped_array.MappedArray`.

```json
${config_doc}
```""")

_settings['mapped_array'] = mapped_array

orders = dict(
    stats=Config(  # flex
        dict()
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['orders'] = Sub("""Sub-config with settings applied to `vectorbt.portfolio.orders.Orders`.

```json
${config_doc}
```""")

_settings['orders'] = orders

trades = dict(
    stats=Config(  # flex
        dict(
            settings=dict(
                incl_open=False
            ),
            template_mapping=dict(
                incl_open_tags=RepEval("['open', 'closed'] if incl_open else ['closed']")
            )
        )
    ),
    plots=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['trades'] = Sub("""Sub-config with settings applied to `vectorbt.portfolio.trades.Trades`.

```json
${config_doc}
```""")

_settings['trades'] = trades

logs = dict(
    stats=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['logs'] = Sub("""Sub-config with settings applied to `vectorbt.portfolio.logs.Logs`.

```json
${config_doc}
```""")

_settings['logs'] = logs

portfolio = dict(
    call_seq='default',
    init_cash=100.,
    init_position=0.,
    cash_deposits=0.,
    cash_earnings=0.,
    cash_dividends=0.,
    size=np.inf,
    size_type='amount',
    signal_direction='longonly',
    order_direction='both',
    price=np.inf,
    fees=0.,
    fixed_fees=0.,
    slippage=0.,
    reject_prob=0.,
    min_size=1e-8,
    max_size=np.inf,
    size_granularity=np.nan,
    price_area_vio_mode='ignore',
    lock_cash=False,
    allow_partial=True,
    raise_reject=False,
    val_price=np.inf,
    accumulate=False,
    sl_stop=np.nan,
    sl_trail=False,
    tp_stop=np.nan,
    stop_entry_price='close',
    stop_exit_price='stoplimit',
    stop_conflict_mode='exit',
    upon_stop_exit='close',
    upon_stop_update='override',
    signal_priority='stop',
    use_stops=None,
    log=False,
    upon_long_conflict='ignore',
    upon_short_conflict='ignore',
    upon_dir_conflict='ignore',
    upon_opposite_entry='reversereduce',
    cash_sharing=False,
    call_pre_segment=False,
    call_post_segment=False,
    ffill_val_price=True,
    update_value=False,
    fill_pos_record=True,
    track_value=True,
    row_wise=False,
    flexible=False,
    seed=None,
    group_by=None,
    broadcast_kwargs=Config(  # flex
        dict(
            require_kwargs=dict(
                requirements='W'
            )
        )
    ),
    template_mapping=Config(  # flex
        dict()
    ),
    keep_inout_raw=True,
    freq=None,
    attach_call_seq=False,
    holding_base_method='from_signals',
    use_in_outputs=True,
    fillna_close=True,
    trades_type='exittrades',
    stats=Config(  # flex
        dict(
            filters=dict(
                has_year_freq=dict(
                    filter_func=lambda self, metric_settings:
                    metric_settings['year_freq'] is not None,
                    warning_message=Sub("Metric '$metric_name' requires year frequency to be set")
                )
            ),
            settings=dict(
                use_asset_returns=False,
                incl_open=False
            ),
            template_mapping=dict(
                incl_open_tags=RepEval("['open', 'closed'] if incl_open else ['closed']")
            )
        )
    ),
    plots=Config(  # flex
        dict(
            subplots=['orders', 'trade_pnl', 'cum_returns'],
            settings=dict(
                use_asset_returns=False
            )
        )
    )
)
"""_"""

__pdoc__['portfolio'] = Sub("""Sub-config with settings applied to `vectorbt.portfolio.base.Portfolio`.

```json
${config_doc}
```""")

_settings['portfolio'] = portfolio

messaging = dict(
    telegram=Config(  # flex
        dict(
            token=None,
            use_context=True,
            persistence='telegram_bot.pickle',
            defaults=Config(  # flex
                dict()
            ),
            drop_pending_updates=True
        )
    ),
    giphy=dict(
        api_key=None,
        weirdness=5
    ),
)
"""_"""

__pdoc__['messaging'] = Sub("""Sub-config with settings applied across `vectorbt.messaging`.

```json
${config_doc}
```

## python-telegram-bot

Sub-config with settings applied to 
[python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot).

Set `persistence` to string to use as `filename` in `telegram.ext.PicklePersistence`.
For `defaults`, see `telegram.ext.Defaults`. Other settings will be distributed across 
`telegram.ext.Updater` and `telegram.ext.updater.Updater.start_polling`.

## GIPHY

Sub-config with settings applied to 
[GIPHY Translate Endpoint](https://developers.giphy.com/docs/api/endpoint#translate).""")

_settings['messaging'] = messaging

pbar = dict(
    disable=False,
    type='tqdm_auto',
    kwargs=Config(  # flex
        dict()
    )
)
"""_"""

__pdoc__['pbar'] = Sub("""Sub-config with settings applied across `vectorbt.utils.pbar`.

```json
${config_doc}
```""")

_settings['pbar'] = pbar


# ############# Settings config ############# #

class SettingsConfig(Config):
    """Extends `vectorbt.utils.config.Config` for global settings."""

    def load_json_templates(self) -> None:
        """Load templates from JSON files."""
        for template_name in ['light', 'dark', 'seaborn']:
            template = Config(json.loads(pkgutil.get_data(__name__, f"templates/{template_name}.json")))
            self['plotting']['themes'][template_name]['template'] = template

    def register_template(self, theme: str) -> None:
        """Register template of a theme."""
        import plotly.io as pio
        import plotly.graph_objects as go

        pio.templates['vbt_' + theme] = go.layout.Template(self['plotting']['themes'][theme]['template'])

    def register_templates(self) -> None:
        """Register templates of all themes."""
        for theme in self['plotting']['themes']:
            self.register_template(theme)

    def set_theme(self, theme: str) -> None:
        """Set default theme."""
        self.register_template(theme)
        self['plotting']['color_schema'].update(self['plotting']['themes'][theme]['color_schema'])
        self['plotting']['layout']['template'] = 'vbt_' + theme

    def reset_theme(self) -> None:
        """Reset to default theme."""
        self.set_theme('light')

    def substitute_sub_config_docs(self, __pdoc__: dict, to_doc_kwargs) -> None:
        """Substitute templiates in sub-config docs."""
        for k, v in __pdoc__.items():
            if k in self:
                config_doc = self[k].stringify(**to_doc_kwargs.get(k, {}))
                __pdoc__[k] = deep_substitute(v, mapping=dict(config_doc=config_doc), sub_id='__pdoc__')


settings = SettingsConfig(
    _settings,
    reset_dct_copy_kwargs_=dict(
        copy_mode='deep'
    ),
    frozen_keys_=True,
    convert_dicts_=Config,
    as_attrs_=True
)
"""Global settings config.

Combines all sub-configs defined in this module."""

try:
    settings.load_json_templates()
    settings.reset_theme()
except ImportError:
    pass

if 'VBT_SETTINGS_PATH' in os.environ:
    settings.load_update(os.environ['VBT_SETTINGS_PATH'], clear=True)

if 'VBT_SETTINGS_OVERRIDE_PATH' in os.environ:
    settings.load_update(os.environ['VBT_SETTINGS_OVERRIDE_PATH'], clear=False)

try:
    settings.register_templates()
except ImportError:
    pass

settings.make_checkpoint()

settings.substitute_sub_config_docs(
    __pdoc__,
    to_doc_kwargs=dict(
        plotting=dict(
            replace={
                'settings.plotting.themes.light.template': '{ ... templates/light.json ... }',
                'settings.plotting.themes.dark.template': '{ ... templates/dark.json ... }',
                'settings.plotting.themes.seaborn.template': '{ ... templates/seaborn.json ... }'
            },
            path='settings.plotting'
        )
    )
)
