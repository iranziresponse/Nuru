"""Structured error logging: failures worth knowing about go to a
persistent, rotating local log file, not just wherever stdout happens to
be pointed, which only helps if someone is watching that terminal window
at the exact moment something goes wrong.

Optionally forwards to Sentry if SENTRY_DSN is set and the sentry_sdk
package is installed; degrades to local-only file logging otherwise, the
same optional-dependency pattern used for OCR (pytesseract/Tesseract).
"""

import logging
import logging.handlers
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nuru_cache")
LOG_PATH = os.path.join(LOG_DIR, "error.log")

_sentry_ready = None  # None = not yet attempted, True/False once resolved


def _init_sentry(logger):
    global _sentry_ready
    if _sentry_ready is not None:
        return _sentry_ready
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        _sentry_ready = False
        return False
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0)
        _sentry_ready = True
    except ImportError:
        logger.warning("SENTRY_DSN is set but the sentry_sdk package isn't installed; "
                        "errors will only be logged locally. `pip install sentry-sdk` to fix.")
        _sentry_ready = False
    except Exception as exc:
        logger.warning(f"Sentry initialization failed, logging locally only: {exc}")
        _sentry_ready = False
    return _sentry_ready


def get_logger(name="nuru"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured by an earlier call
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Persistent file logging is the whole point of this module, but it
    # must never be the reason a caller's own error-handling path itself
    # raises: fall back to console-only if the directory can't be created
    # or the file can't be opened (permissions, an unwritable path, etc).
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    _init_sentry(logger)
    return logger


def report_exception(exc, **context):
    """Log an exception with whatever context the caller has (a filename,
    a token, an action) and forward it to Sentry if configured. Never
    raises: a failure to report an error must not itself become one."""
    logger = get_logger()
    try:
        logger.error(f"{exc} | {context}", exc_info=exc)
    except Exception:
        pass
    if _sentry_ready:
        try:
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                for key, value in context.items():
                    scope.set_extra(key, value)
                sentry_sdk.capture_exception(exc)
        except Exception:
            pass
