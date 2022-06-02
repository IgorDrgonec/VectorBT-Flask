# Copyright (c) 2022 Oleg Polakow. All rights reserved.

"""Base class for working with records that can make use of OHLC data."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb
from vectorbtpro.records.base import Records
from vectorbtpro.utils import checks

PriceRecordsT = tp.TypeVar("PriceRecordsT", bound="PriceRecords")


class PriceRecords(Records):
    """Extends `vectorbtpro.records.base.Records` for records that can make use of OHLC data."""

    @classmethod
    def from_records(
        cls: tp.Type[PriceRecordsT],
        wrapper: ArrayWrapper,
        records: tp.RecordArray,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        attach_price: bool = True,
        **kwargs,
    ) -> PriceRecordsT:
        """Build `PriceRecords` from records."""
        return cls(
            wrapper,
            records,
            open=open if attach_price else None,
            high=high if attach_price else None,
            low=low if attach_price else None,
            close=close if attach_price else None,
            **kwargs,
        )

    @classmethod
    def resolve_row_stack_kwargs(
        cls: tp.Type[PriceRecordsT],
        *objs: tp.MaybeTuple[PriceRecordsT],
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `PriceRecords` after stacking along columns."""
        kwargs = Records.resolve_row_stack_kwargs(*objs, **kwargs)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PriceRecords):
                raise TypeError("Each object to be merged must be an instance of PriceRecords")
        for price_name in ("open", "high", "low", "close"):
            if price_name not in kwargs:
                price_objs = []
                stack_price_objs = True
                for obj in objs:
                    if getattr(obj, price_name) is not None:
                        price_objs.append(getattr(obj, price_name))
                    else:
                        stack_price_objs = False
                        break
                if stack_price_objs:
                    kwargs[price_name] = kwargs["wrapper"].row_stack_and_wrap(*price_objs, group_by=False)
        return kwargs

    @classmethod
    def resolve_column_stack_kwargs(
        cls: tp.Type[PriceRecordsT],
        *objs: tp.MaybeTuple[PriceRecordsT],
        reindex_kwargs: tp.KwargsLike = None,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `PriceRecords` after stacking along columns."""
        kwargs = Records.resolve_column_stack_kwargs(*objs, reindex_kwargs=reindex_kwargs, **kwargs)
        kwargs.pop("reindex_kwargs", None)
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, PriceRecords):
                raise TypeError("Each object to be merged must be an instance of PriceRecords")
        for price_name in ("open", "high", "low", "close"):
            if price_name not in kwargs:
                price_objs = []
                stack_price_objs = True
                for obj in objs:
                    if getattr(obj, "_" + price_name) is not None:
                        price_objs.append(getattr(obj, price_name))
                    else:
                        stack_price_objs = False
                        break
                if stack_price_objs:
                    new_price = kwargs["wrapper"].column_stack_and_wrap(
                        *price_objs,
                        reindex_kwargs=reindex_kwargs,
                        group_by=False,
                    )
                    if price_name == "close":
                        if fbfill_close:
                            new_price = new_price.vbt.fbfill()
                        elif ffill_close:
                            new_price = new_price.vbt.ffill()
                    kwargs[price_name] = new_price
        return kwargs

    def __init__(
        self,
        wrapper: ArrayWrapper,
        records_arr: tp.RecordArray,
        open: tp.Optional[tp.ArrayLike] = None,
        high: tp.Optional[tp.ArrayLike] = None,
        low: tp.Optional[tp.ArrayLike] = None,
        close: tp.Optional[tp.ArrayLike] = None,
        **kwargs,
    ) -> None:
        Records.__init__(
            self,
            wrapper,
            records_arr,
            open=open,
            high=high,
            low=low,
            close=close,
            **kwargs,
        )
        self._open = open
        self._high = high
        self._low = low
        self._close = close

    def indexing_func(self: PriceRecordsT, *args, records_meta: tp.DictLike = None, **kwargs) -> PriceRecordsT:
        """Perform indexing on `PriceRecords`."""
        if records_meta is None:
            records_meta = Records.indexing_func_meta(self, *args, **kwargs)
        prices = {}
        for price_name in ("open", "high", "low", "close"):
            if getattr(self, "_" + price_name) is not None:
                new_price = to_2d_array(getattr(self, "_" + price_name))
                if new_price.shape[0] > 1:
                    new_price = new_price[records_meta["wrapper_meta"]["row_idxs"], :]
                if new_price.shape[1] > 1:
                    new_price = new_price[:, records_meta["wrapper_meta"]["col_idxs"]]
            else:
                new_price = None
            prices[price_name] = new_price
        return self.replace(
            wrapper=records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=records_meta["new_records_arr"],
            open=prices["open"],
            high=prices["high"],
            low=prices["low"],
            close=prices["close"],
        )

    def resample(
        self: PriceRecordsT,
        *args,
        ffill_close: bool = False,
        fbfill_close: bool = False,
        records_meta: tp.DictLike = None,
        **kwargs,
    ) -> PriceRecordsT:
        """Perform resampling on `PriceRecords`."""
        if records_meta is None:
            records_meta = self.resample_meta(*args, **kwargs)
        if self._open is None:
            new_open = None
        else:
            new_open = self.open.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.first_reduce_nb,
            )
        if self._high is None:
            new_high = None
        else:
            new_high = self.high.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.max_reduce_nb,
            )
        if self._low is None:
            new_low = None
        else:
            new_low = self.low.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.min_reduce_nb,
            )
        if self._close is None:
            new_close = None
        else:
            new_close = self.close.vbt.resample_apply(
                records_meta["wrapper_meta"]["resampler"],
                nb.last_reduce_nb,
            )
            if fbfill_close:
                new_close = new_close.vbt.fbfill()
            elif ffill_close:
                new_close = new_close.vbt.ffill()
        return self.replace(
            wrapper=records_meta["wrapper_meta"]["new_wrapper"],
            records_arr=records_meta["new_records_arr"],
            open=new_open,
            high=new_high,
            low=new_low,
            close=new_close,
        )

    @property
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Open price."""
        if self._open is None:
            return None
        return self.wrapper.wrap(self._open, group_by=False)

    @property
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """High price."""
        if self._high is None:
            return None
        return self.wrapper.wrap(self._high, group_by=False)

    @property
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Low price."""
        if self._low is None:
            return None
        return self.wrapper.wrap(self._low, group_by=False)

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """Close price."""
        if self._close is None:
            return None
        return self.wrapper.wrap(self._close, group_by=False)
