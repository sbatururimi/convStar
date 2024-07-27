import logging
from rich.logging import RichHandler
from logging.handlers import TimedRotatingFileHandler

def get_logger(name:str, log_filename:str = None) -> logging.Logger:
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - [%(funcName)s] - %(name)s - %(levelname)s - %(message)s"
    )

    # Get the logger
    root_logger = logging.getLogger()
    root_logger.handlers = []

    # Get your logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Disable propagation to prevent the root logger from also handling log messages
    logger.propagate = False

    # Remove all existing handlers
    while logger.handlers:
        logger.handlers.pop()

    # Create and set up the RichHandler
    rich_handler = RichHandler(rich_tracebacks=True)
    rich_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(rich_handler)

    if log_filename:
        handler = TimedRotatingFileHandler(
            log_filename, when="midnight", interval=1, backupCount=2
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
