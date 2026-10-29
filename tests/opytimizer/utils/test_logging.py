from opytimizer.utils import logging


def test_logging_to_file():
    logger = logging.get_logger(__name__)

    assert logger.to_file("msg") is None


def test_logging_get_console_handler():
    c = logging.get_console_handler()

    assert c is not None


def test_logging_get_timed_file_handler():
    f = logging.get_timed_file_handler()

    assert f is not None


def test_logging_get_logger():
    logger = logging.get_logger(__name__)

    assert logger.name == "test_logging"

    assert logger.hasHandlers() is True


def test_logging_set_logging():
    logger = logging.get_logger("opytimizer.test_set_logging")

    try:
        logging.set_logging(False)

        assert logging.is_logging_enabled() is False
        assert logger.disabled is True
        # Loggers created while disabled are born disabled
        assert logging.get_logger("opytimizer.test_set_logging_new").disabled is True
    finally:
        logging.set_logging(True)

    assert logging.is_logging_enabled() is True
    assert logger.disabled is False


def test_logging_disabled():
    with logging.disabled():
        assert logging.is_logging_enabled() is False

    assert logging.is_logging_enabled() is True

    with logging.disabled(False):
        assert logging.is_logging_enabled() is True

    # Never re-enables a logging system that was globally turned off
    try:
        logging.set_logging(False)
        with logging.disabled(False):
            assert logging.is_logging_enabled() is False
    finally:
        logging.set_logging(True)
