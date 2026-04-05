"""Logging configuration for eostrata.

Sets up a daily-rotating file handler alongside the console handler.
Rotation happens at midnight; files older than 30 days are deleted
automatically by the handler.

The log file path is controlled by ``EOSTRATA_LOG_FILE`` (default:
``data/eostrata.log``).  Set it to an empty string to disable file logging.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

# Endpoints polled on a tight loop by the browser (job status, store usage).
# Suppress their access-log entries to avoid filling logs with noise.
_POLLING_PATHS = frozenset(
    [
        "/processes/jobs",
        "/store-usage",
    ]
)


class _SuppressPollingFilter(logging.Filter):
    """Drop uvicorn access-log records for high-frequency polling endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in _POLLING_PATHS)


def setup_logging(
    verbose: bool = False,
    log_file: Path | str | None = None,
    *,
    rich_console: bool = True,
) -> None:
    """Configure root logger with a console handler and an optional rotating file handler.

    Parameters
    ----------
    verbose:
        If True, set log level to DEBUG; otherwise INFO.
    log_file:
        Path to the log file.  Pass ``None`` to use the value from settings.
        Pass an empty string or ``Path("")`` to disable file logging entirely.
    rich_console:
        If True (default), attach a Rich console handler for terminal output.
        Set to False when attaching to a server that manages its own console output.
    """
    from eostrata.config import settings

    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()

    # Avoid adding duplicate handlers if called more than once
    if root.handlers:
        return

    root.setLevel(level)

    if rich_console:
        from rich.logging import RichHandler

        root.addHandler(RichHandler(rich_tracebacks=True, show_path=False))

    # Resolve log file path
    if log_file is None:
        log_file = settings.log_file
    log_path = Path(log_file) if log_file else None

    # Route warnings.warn() calls through the logging system so they appear in
    # the file alongside regular log messages.
    logging.captureWarnings(True)

    if log_path and str(log_path) != "":
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path,
            when="midnight",
            backupCount=30,  # keep 30 days of rotated files
            encoding="utf-8",
            utc=True,
        )
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            )
        )
        root.addHandler(file_handler)

        # Uvicorn's dictConfig resets its loggers and disables propagation to the
        # root logger, so the file handler would never receive uvicorn records.
        # Attach it directly to the uvicorn loggers to ensure access logs and
        # uvicorn-level messages land in the file.
        for uvicorn_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
            logging.getLogger(uvicorn_logger_name).addHandler(file_handler)

        logging.getLogger(__name__).debug("File logging enabled: %s", log_path)

    # Suppress high-frequency polling requests from the access log regardless
    # of whether file logging is enabled (applies to the console too).
    logging.getLogger("uvicorn.access").addFilter(_SuppressPollingFilter())
