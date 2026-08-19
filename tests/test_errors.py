import logging

import pytest

import errors


@pytest.fixture(autouse=True)
def isolated_logger(tmp_path, monkeypatch):
    """Every test gets a fresh, unconfigured logger writing to a throwaway
    file, isolated from the real .nuru_cache/error.log."""
    monkeypatch.setattr(errors, "LOG_PATH", str(tmp_path / "error.log"))
    monkeypatch.setattr(errors, "_sentry_ready", None)
    logging.getLogger("nuru").handlers.clear()
    yield
    logging.getLogger("nuru").handlers.clear()


def test_get_logger_writes_to_the_configured_file(tmp_path):
    logger = errors.get_logger()
    logger.warning("hello from a test")

    log_path = tmp_path / "error.log"
    assert log_path.exists()
    assert "hello from a test" in log_path.read_text(encoding="utf-8")


def test_get_logger_is_idempotent_and_does_not_duplicate_handlers():
    logger1 = errors.get_logger()
    handler_count = len(logger1.handlers)
    logger2 = errors.get_logger()
    assert logger1 is logger2
    assert len(logger2.handlers) == handler_count


def test_report_exception_includes_context_and_never_raises(tmp_path):
    try:
        raise ValueError("boom")
    except ValueError as exc:
        errors.report_exception(exc, filename="a.pdf", action="scan")  # must not raise

    log_text = (tmp_path / "error.log").read_text(encoding="utf-8")
    assert "boom" in log_text
    assert "a.pdf" in log_text
    assert "ValueError" in log_text  # traceback included


def test_report_exception_survives_a_broken_log_path(monkeypatch):
    monkeypatch.setattr(errors, "LOG_PATH", "Z:\\this\\path\\cannot\\possibly\\exist\\error.log")
    logging.getLogger("nuru").handlers.clear()
    try:
        raise RuntimeError("should not crash the caller")
    except RuntimeError as exc:
        errors.report_exception(exc)  # must not raise even if file logging setup fails


def test_sentry_not_configured_by_default(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    logger = errors.get_logger()
    assert errors._init_sentry(logger) is False


def test_sentry_gracefully_skips_when_dsn_set_but_package_missing(monkeypatch):
    """A None entry in sys.modules is the standard way to make `import x`
    raise ImportError regardless of whether x is actually installed —
    more reliable than monkeypatching __import__ itself, which doesn't
    reliably intercept an already-imported module's import statement."""
    import sys
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    monkeypatch.setenv("SENTRY_DSN", "https://fake@fake.ingest.sentry.io/1")
    logger = errors.get_logger()
    assert errors._init_sentry(logger) is False
