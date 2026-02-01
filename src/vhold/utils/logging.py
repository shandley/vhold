"""Logging configuration for vhold using loguru."""

import sys
from pathlib import Path

from loguru import logger

# Remove default handler
logger.remove()

# Global state
_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    show_time: bool = True,
) -> None:
    """Configure loguru logging for vhold.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        show_time: Whether to show timestamps
    """
    global _configured

    if _configured:
        return

    # Format string
    if show_time:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    else:
        fmt = (
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        )

    # Console handler
    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=True,
    )

    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            format=fmt,
            level="DEBUG",  # Always log everything to file
            rotation="10 MB",
            retention="1 week",
        )

    _configured = True


def get_logger(name: str = "vhold"):
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger bound to the given name
    """
    return logger.bind(name=name)


# Simple logging interface
def log_info(message: str) -> None:
    """Log an info message."""
    logger.info(message)


def log_warning(message: str) -> None:
    """Log a warning message."""
    logger.warning(message)


def log_error(message: str) -> None:
    """Log an error message."""
    logger.error(message)


def log_debug(message: str) -> None:
    """Log a debug message."""
    logger.debug(message)
