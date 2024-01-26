# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for constructing and displaying figures."""

from vectorbtpro.utils.module_ import assert_can_import

assert_can_import("plotly")

from pathlib import Path

import pandas as pd

from plotly.graph_objects import Figure as _Figure, FigureWidget as _FigureWidget
from plotly.subplots import make_subplots as _make_subplots

from vectorbtpro import _typing as tp
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.datetime_ import get_rangebreaks
from vectorbtpro.utils.path_ import check_mkdir

__all__ = [
    "Figure",
    "FigureWidget",
    "make_figure",
    "make_subplots",
]


def resolve_axis_refs(
    add_trace_kwargs: tp.KwargsLike = None,
    xref: tp.Optional[str] = None,
    yref: tp.Optional[str] = None,
) -> tp.Tuple[str, str]:
    """Get x-axis and y-axis references."""
    if add_trace_kwargs is None:
        add_trace_kwargs = {}
    row = add_trace_kwargs.get("row", 1)
    col = add_trace_kwargs.get("col", 1)
    if xref is None:
        if col == 1:
            xref = "x"
        else:
            xref = "x" + str(col)
    if yref is None:
        if row == 1:
            yref = "y"
        else:
            yref = "y" + str(row)
    return xref, yref


def get_domain(ref: str, fig: tp.BaseFigure) -> tp.Tuple[int, int]:
    """Get domain of a coordinate axis."""
    axis = ref[0] + "axis" + ref[1:]
    if axis in fig.layout:
        if "domain" in fig.layout[axis]:
            if fig.layout[axis]["domain"] is not None:
                return fig.layout[axis]["domain"]
    return 0, 1


FigureMixinT = tp.TypeVar("FigureMixinT", bound="FigureMixin")


class FigureMixin:
    """Mixin class for figures."""

    def auto_rangebreaks(self: FigureMixinT, index: tp.Optional[tp.IndexLike] = None, **kwargs) -> FigureMixinT:
        """Set range breaks automatically based on `vectorbtpro.utils.datetime_.get_rangebreaks`.

        Changes the figure in place and returns it."""
        if index is None:
            for d in self.data:
                if "x" in d:
                    d_index = pd.Index(self.data[0].x)
                    if not isinstance(d_index, pd.DatetimeIndex):
                        return self
                    if index is None:
                        index = d_index
                    elif not index.equals(d_index):
                        index = index.union(d_index)
            if index is None:
                raise ValueError("Couldn't extract x-axis values, please provide index")
        rangebreaks = get_rangebreaks(index, **kwargs)
        return self.update_xaxes(rangebreaks=rangebreaks)

    def skip_index(self: FigureMixinT, skip_index: tp.IndexLike) -> FigureMixinT:
        """Skip index values.

        Changes the figure in place and returns it."""
        return self.update_xaxes(rangebreaks=[dict(values=skip_index)])

    def resolve_show_args(
        self,
        *args,
        auto_rangebreaks: tp.Union[None, bool, dict] = None,
        **kwargs,
    ) -> tp.Tuple[tp.Args, tp.Kwargs]:
        """Display the figure."""
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        _self = self
        if auto_rangebreaks is None:
            auto_rangebreaks = plotting_cfg["auto_rangebreaks"]
        if auto_rangebreaks not in (False, None):
            if auto_rangebreaks is True:
                _self.auto_rangebreaks()
            elif isinstance(auto_rangebreaks, dict):
                _self.auto_rangebreaks(**auto_rangebreaks)
            else:
                raise TypeError("Argument auto_rangebreaks must be either bool or dict")
        pre_show_func = plotting_cfg.get("pre_show_func", None)
        if pre_show_func is not None:
            __self = pre_show_func(_self)
            if __self is not None:
                _self = __self
        fig_kwargs = dict(width=_self.layout.width, height=_self.layout.height)
        kwargs = merge_dicts(fig_kwargs, plotting_cfg["show_kwargs"], kwargs)
        return args, kwargs

    def show(self, *args, **kwargs) -> None:
        """Display the figure."""
        raise NotImplementedError

    def show_png(self, **kwargs) -> None:
        """Display the figure in PNG format."""
        self.show(renderer="png", **kwargs)

    def show_svg(self, **kwargs) -> None:
        """Display the figure in SVG format."""
        self.show(renderer="svg", **kwargs)

    def save_svg_for_docs(
        self,
        figure_name: str,
        dir_path: tp.PathLike = Path("./svg"),
        mkdir_kwargs: tp.KwargsLike = None,
        show: bool = True,
        show_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        """Save the figure in both light and dark SVG format for documentation."""
        if not isinstance(dir_path, Path):
            dir_path = Path(dir_path)
        if mkdir_kwargs is None:
            mkdir_kwargs = {}
        if "mkdir" not in mkdir_kwargs:
            mkdir_kwargs["mkdir"] = True
        check_mkdir(dir_path, **mkdir_kwargs)
        self.update_layout(template="vbt_light")
        self.write_image(dir_path / (figure_name + ".light.svg"), **kwargs)
        self.update_layout(template="vbt_dark")
        self.write_image(dir_path / (figure_name + ".dark.svg"), **kwargs)
        if show:
            if show_kwargs is None:
                show_kwargs = {}
            self.show_svg(**show_kwargs)


class Figure(_Figure, FigureMixin):
    """Figure.

    Extends `plotly.graph_objects.Figure`."""

    def __init__(self, *args, **kwargs) -> None:
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        layout = kwargs.pop("layout", {})
        super().__init__(*args, **kwargs)
        self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

    def show(self, *args, **kwargs) -> None:
        args, kwargs = self.resolve_show_args(*args, **kwargs)
        _Figure.show(self, *args, **kwargs)


class FigureWidget(_FigureWidget, FigureMixin):
    """Figure widget.

    Extends `plotly.graph_objects.FigureWidget`."""

    def __init__(self, *args, **kwargs) -> None:
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        layout = kwargs.pop("layout", {})
        super().__init__(*args, **kwargs)
        self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

    def show(self, *args, **kwargs) -> None:
        args, kwargs = self.resolve_show_args(*args, **kwargs)
        _FigureWidget.show(self, *args, **kwargs)


try:
    from plotly_resampler import FigureResampler as _FigureResampler, FigureWidgetResampler as _FigureWidgetResampler

    class FigureResampler(_FigureResampler, FigureMixin):
        """Figure resampler.

        Extends `plotly.graph_objects.Figure`."""

        def __init__(self, *args, **kwargs) -> None:
            from vectorbtpro._settings import settings

            plotting_cfg = settings["plotting"]

            layout = kwargs.pop("layout", {})
            super().__init__(*args, **kwargs)
            self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

        def show(self, *args, **kwargs) -> None:
            args, kwargs = self.resolve_show_args(*args, **kwargs)
            _FigureResampler.show(self, *args, **kwargs)

    class FigureWidgetResampler(_FigureWidgetResampler, FigureMixin):
        """Figure widget resampler.

        Extends `plotly.graph_objects.FigureWidget`."""

        def __init__(self, *args, **kwargs) -> None:
            from vectorbtpro._settings import settings

            plotting_cfg = settings["plotting"]

            layout = kwargs.pop("layout", {})
            super().__init__(*args, **kwargs)
            self.update_layout(**merge_dicts(plotting_cfg["layout"], layout))

        def show(self, *args, **kwargs) -> None:
            args, kwargs = self.resolve_show_args(*args, **kwargs)
            _FigureWidgetResampler.show(self, *args, **kwargs)

except ImportError:
    FigureResampler = Figure
    FigureWidgetResampler = FigureWidget


def make_figure(
    *args,
    use_widgets: tp.Optional[bool] = None,
    use_resampler: tp.Optional[bool] = None,
    **kwargs,
) -> tp.BaseFigure:
    """Make a new Plotly figure.

    If `use_widgets` is True, returns `FigureWidget`, otherwise `Figure`.

    If `use_resampler` is True, additionally wraps the class using `plotly_resampler`.

    Defaults are defined under `vectorbtpro._settings.plotting`."""
    from vectorbtpro._settings import settings

    plotting_cfg = settings["plotting"]

    if use_widgets is None:
        use_widgets = plotting_cfg["use_widgets"]
    if use_resampler is None:
        use_resampler = plotting_cfg["use_resampler"]

    if use_widgets:
        if use_resampler is None:
            return FigureWidgetResampler(*args, **kwargs)
        if use_resampler:
            assert_can_import("plotly_resampler")
            return FigureWidgetResampler(*args, **kwargs)
        return FigureWidget(*args, **kwargs)
    if use_resampler is None:
        return FigureResampler(*args, **kwargs)
    if use_resampler:
        assert_can_import("plotly_resampler")
        return FigureResampler(*args, **kwargs)
    return Figure(*args, **kwargs)


def make_subplots(
    *args,
    use_widgets: tp.Optional[bool] = None,
    use_resampler: tp.Optional[bool] = None,
    **kwargs,
) -> tp.BaseFigure:
    """Make Plotly subplots using `make_figure`."""
    return make_figure(_make_subplots(*args, **kwargs), use_widgets=use_widgets, use_resampler=use_resampler)
