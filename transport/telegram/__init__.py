"""
Telegram Transport Module

Pure I/O layer for Telegram messaging.
Exports: TelegramTransport, NormalizedMessage, create_telegram_transport
"""

from transport.telegram.transport import (
    TelegramTransport,
    TelegramMessage,
    NormalizedMessage,
    create_telegram_transport,
)

__all__ = [
    "TelegramTransport",
    "TelegramMessage",
    "NormalizedMessage",
    "create_telegram_transport",
]
