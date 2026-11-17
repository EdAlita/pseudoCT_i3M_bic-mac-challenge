"""
logger.py

This script provides logging utilities, including the function to generate
loggers with customizable logging levels and formats. It supports logging
to both files and the console with colorized output depending on the log level.

Dependencies:
- logging: Standard Python logging library.
- pathlib: Provides the Path class for file system paths.
- tqdm: Library for progress bars.
- os: Standard Python library for OS interactions.
- termcolor: For colorizing log messages by level.

Usage:
Import the logger generation function and call it with a name to get a configured logger instance.

Owner:
Edwing Ulin

Version:
v1.0.0
"""

import logging
from pathlib import Path
from tqdm import tqdm
import os
from datetime import datetime
from termcolor import colored

def logger_gen(name: str) -> logging.Logger:
    """
    Generates a logger with the specified name, log level, and optional log file.

    Args:
    -----
        name (str): The name of the logger. Typically, the module or script name.

    Returns:
    --------
        logging.Logger: A logger instance configured with file and stream handlers.
    """
    cwd = Path(os.getcwd())
    logs_dir = cwd / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_handler = logging.FileHandler(logs_dir / f'{name}_{timestamp}.log', mode='a')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setStream(tqdm)
    stream_handler.setFormatter(ColorFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

    logger.setLevel(logging.DEBUG)

    return logger


def colorize_log(record: logging.LogRecord) -> str:
    """
    Colorizes the log message based on the log level.

    Args:
    -----
        record (logging.LogRecord): The log record containing message and level.

    Returns:
    --------
        str: The colorized log message.
    """
    log_message = record.getMessage()

    if record.levelno == logging.WARNING:
        return colored(log_message, 'yellow')  # Yellow for warnings
    elif record.levelno == logging.ERROR:
        return colored(log_message, 'red')  # Red for errors

    return log_message  # Default color for other logs


class ColorFormatter(logging.Formatter):
    """
    A custom logging formatter that applies colorization to the log message based on its level.
    """

    def format(self, record: logging.LogRecord) -> str:
        record.msg = colorize_log(record)
        return super().format(record)
