"""Logging-based methods and helpers.
"""

import logging
import sys
from contextlib import contextmanager
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
from typing import Generator, List

FORMATTER = logging.Formatter("%(asctime)s - %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "opytimizer.log"
LOG_LEVEL = logging.DEBUG

# Prefix shared by every logger created through `get_logger`
LOGGER_PREFIX = "opytimizer"

# Global switch read by `get_logger`, so loggers created after a call to
# `set_logging(False)` (e.g., lazily imported modules) are born disabled
_ENABLED = True


class Logger(logging.Logger):
    """A customized Logger file that enables the possibility of only logging to file."""

    def to_file(self, msg: str, *args, **kwargs) -> None:
        """Logs the message only to the logging file.

        Args:
            msg: Message to be logged.

        """

        # Avoids toggling the console handler level when nothing would be emitted
        if not self.isEnabledFor(logging.INFO):
            return

        self.handlers[0].setLevel(logging.CRITICAL)
        self.info(msg, *args, **kwargs)
        self.handlers[0].setLevel(LOG_LEVEL)


def _opytimizer_loggers() -> List[logging.Logger]:
    """Gets every logger already created by the library.

    Returns:
        (List[logging.Logger]): Loggers whose name starts with `LOGGER_PREFIX`.

    """

    return [
        logger
        for name, logger in logging.Logger.manager.loggerDict.items()
        if isinstance(logger, logging.Logger) and name.startswith(LOGGER_PREFIX)
    ]


def set_logging(enabled: bool) -> None:
    """Enables or disables the whole logging system of the library.

    When disabled, no message reaches the console nor the logging file, which
    avoids the disk writes of the file handler during the optimization.

    Args:
        enabled: Whether the library's loggers should emit messages.

    """

    global _ENABLED
    _ENABLED = bool(enabled)

    for logger in _opytimizer_loggers():
        logger.disabled = not _ENABLED


def is_logging_enabled() -> bool:
    """Checks whether the library's logging system is enabled.

    Returns:
        (bool): True if the loggers are emitting messages.

    """

    return _ENABLED


@contextmanager
def disabled(disable: bool = True) -> Generator[None, None, None]:
    """Temporarily disables the library's logging system.

    The previous state is restored on exit. With `disable=False` it is a no-op,
    so it never re-enables a logging system that was globally turned off.

    Args:
        disable: Whether logging should be disabled inside the context.

    """

    if not disable:
        yield
        return

    previous = is_logging_enabled()
    set_logging(False)
    try:
        yield
    finally:
        set_logging(previous)


def get_console_handler() -> StreamHandler:
    """Gets a console handler to handle logging into console.

    Returns:
        (StreamHandler): Handler to output information into console.

    """

    console_handler = StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)

    return console_handler


def get_timed_file_handler() -> TimedRotatingFileHandler:
    """Gets a timed file handler to handle logging into files.

    Returns:
        (TimedRotatingFileHandler): Handler to output information into timed files.

    """

    file_handler = TimedRotatingFileHandler(LOG_FILE, delay=True, when="midnight")
    file_handler.setFormatter(FORMATTER)

    return file_handler


def get_logger(logger_name: str) -> Logger:
    """Gets a logger and make it avaliable for further use.

    Args:
        logger_name: The name of the logger.

    Returns:
        (Logger): Logger instance.

    """

    logging.setLoggerClass(Logger)

    logger = logging.getLogger(logger_name)

    logger.setLevel(LOG_LEVEL)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_timed_file_handler())
    logger.propagate = False
    logger.disabled = not _ENABLED

    return logger
