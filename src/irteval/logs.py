"""Console + file logging for long experiment runs."""
import logging
import sys


def get_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger.info
