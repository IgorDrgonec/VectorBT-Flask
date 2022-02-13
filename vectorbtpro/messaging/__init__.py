# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Modules for messaging."""

from vectorbtpro.utils.module_ import create__all__

__blacklist__ = []

try:
    import telegram
except ImportError:
    __blacklist__.append("telegram")
else:
    from vectorbtpro.messaging.telegram import TelegramBot


__all__ = create__all__(__name__)
__pdoc__ = {k: False for k in __all__}
