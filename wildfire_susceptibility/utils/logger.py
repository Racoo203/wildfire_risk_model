import logging
from pathlib import Path
from typing import Union, Optional

def setup_logger(
    name: str = "wildfire_risk_model",
    log_file: Union[str, Path] = "pipeline.log",
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure and return a logger with both file and console handlers.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        return logger

    log_file = Path(log_file)
    log_file.parent.mkdir(parents = True, exist_ok = True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_formatter = logging.Formatter(
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        fmt = "%(levelname)s: %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized: {name}")
    return logger