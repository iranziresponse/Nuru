import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import automation


# ---- webhook ------------------------------------------------------------

class _CatchHandler(BaseHTTPRequestHandler):
    received = {}

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        _CatchHandler.received["body"] = json.loads(body)
        _CatchHandler.received["content_type"] = self.headers["Content-Type"]
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def local_receiver():
    server = HTTPServer(("127.0.0.1", 0), _CatchHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/hook"
    thread.join(timeout=2)


def test_send_webhook_posts_json_payload(local_receiver):
    ok, message = automation.send_webhook(local_receiver, {"Vendor": "Acme", "Total": "$10.00"})
    assert ok is True
    assert _CatchHandler.received["body"] == {"Vendor": "Acme", "Total": "$10.00"}
    assert _CatchHandler.received["content_type"] == "application/json"


def test_send_webhook_rejects_invalid_url():
    ok, message = automation.send_webhook("not-a-url", {"a": 1})
    assert ok is False
    assert "valid URL" in message


def test_send_webhook_handles_connection_refused():
    ok, message = automation.send_webhook("http://127.0.0.1:1/nope", {"a": 1})
    assert ok is False
    assert "reach" in message.lower()


# ---- SSRF guard -----------------------------------------------------------

def test_link_local_and_metadata_endpoint_always_blocked():
    assert automation._webhook_host_is_safe("169.254.169.254") is False  # cloud metadata
    assert automation._webhook_host_is_safe("224.0.0.1") is False  # multicast


def test_loopback_and_private_allowed_by_default_but_blockable():
    assert automation._webhook_host_is_safe("127.0.0.1") is True
    assert automation._webhook_host_is_safe("10.0.0.5") is True


def test_public_only_mode_blocks_loopback_and_private(monkeypatch):
    monkeypatch.setenv("NURU_WEBHOOK_PUBLIC_ONLY", "true")
    assert automation._webhook_host_is_safe("127.0.0.1") is False
    assert automation._webhook_host_is_safe("10.0.0.5") is False


def test_send_webhook_rejects_metadata_endpoint():
    ok, message = automation.send_webhook("http://169.254.169.254/latest/meta-data/", {"a": 1})
    assert ok is False
    assert "isn't allowed" in message.lower()


def test_send_webhook_does_not_follow_a_redirect():
    """The redirect target here is a perfectly safe local server. The
    point is to isolate "redirects are never followed" from "unsafe
    targets are blocked" (covered separately above), by directly checking
    the target never received a request rather than matching error text,
    which is sensitive to exact timing/exception-path details."""
    target_hits = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            target_hits.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    target_server = HTTPServer(("127.0.0.1", 0), TargetHandler)
    target_port = target_server.server_address[1]
    target_thread = threading.Thread(target=target_server.handle_request, daemon=True)
    target_thread.start()

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/should-not-be-reached")
            self.end_headers()

        def log_message(self, *args):
            pass

    redirect_server = HTTPServer(("127.0.0.1", 0), RedirectHandler)
    redirect_port = redirect_server.server_address[1]
    redirect_thread = threading.Thread(target=redirect_server.handle_request, daemon=True)
    redirect_thread.start()

    ok, message = automation.send_webhook(f"http://127.0.0.1:{redirect_port}/redir", {"a": 1})
    redirect_thread.join(timeout=2)

    assert ok is False
    assert target_hits == []  # the redirect must never have been followed
    # target_thread is a daemon thread; if the target was (correctly) never
    # hit, it just stays blocked in handle_request() until the process exits.


# ---- email ---------------------------------------------------------------

def test_send_email_reports_friendly_message_when_unconfigured(monkeypatch):
    for var in ("NURU_SMTP_HOST", "NURU_SMTP_USER", "NURU_SMTP_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    ok, message = automation.send_email("someone@example.com", "Subject", [{"label": "A", "value": "B"}])
    assert ok is False
    assert "isn't set up" in message


def test_send_email_rejects_bad_address(monkeypatch):
    monkeypatch.setenv("NURU_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("NURU_SMTP_USER", "bot@example.com")
    monkeypatch.setenv("NURU_SMTP_PASSWORD", "secret")
    ok, message = automation.send_email("not-an-email", "Subject", [])
    assert ok is False
    assert "valid email" in message


def test_send_email_uses_configured_smtp_client(monkeypatch):
    monkeypatch.setenv("NURU_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("NURU_SMTP_USER", "bot@example.com")
    monkeypatch.setenv("NURU_SMTP_PASSWORD", "secret")

    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"], sent["port"] = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            sent["tls"] = True

        def login(self, user, password):
            sent["login"] = (user, password)

        def send_message(self, message):
            sent["to"] = message["To"]
            sent["subject"] = message["Subject"]
            sent["body"] = message.get_content()

    monkeypatch.setattr(automation.smtplib, "SMTP", FakeSMTP)

    ok, message = automation.send_email(
        "ap@company.com", "Invoice from Acme",
        [{"label": "Supplier Name", "value": "Acme"}, {"label": "Total", "value": "$10.00"}],
    )
    assert ok is True
    assert sent["to"] == "ap@company.com"
    assert sent["subject"] == "Invoice from Acme"
    assert "Supplier Name: Acme" in sent["body"]
    assert sent["login"] == ("bot@example.com", "secret")


# ---- archive ---------------------------------------------------------------

def test_build_archive_filename_uses_recognized_label_and_cleans_characters():
    rows = [{"label": "Supplier Name", "value": "Acme / Supply Co!"},
            {"label": "Transaction Date", "value": "2026-04-02"}]
    name = automation.build_archive_filename(rows, "source.pdf")
    assert name == "Acme_Supply_Co_2026-04-02.pdf"


def test_build_archive_filename_falls_back_when_fields_missing():
    name = automation.build_archive_filename([], "original name.pdf")
    assert name == "Document.pdf"


def test_archive_file_moves_and_renames(tmp_path, monkeypatch):
    src = tmp_path / "source.pdf"
    src.write_bytes(b"%PDF-fake")
    dest_dir = tmp_path / "dest"
    monkeypatch.setenv("NURU_ARCHIVE_ALLOWED_ROOTS", str(dest_dir))

    ok, message = automation.archive_file(str(src), str(dest_dir), "Acme_2026-04-02.pdf")
    assert ok is True
    assert (dest_dir / "Acme_2026-04-02.pdf").exists()
    assert not src.exists()


def test_archive_file_avoids_overwriting_existing_file(tmp_path, monkeypatch):
    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    (dest_dir / "Acme.pdf").write_bytes(b"already here")
    monkeypatch.setenv("NURU_ARCHIVE_ALLOWED_ROOTS", str(dest_dir))

    src = tmp_path / "source.pdf"
    src.write_bytes(b"%PDF-fake")

    ok, message = automation.archive_file(str(src), str(dest_dir), "Acme.pdf")
    assert ok is True
    assert (dest_dir / "Acme (2).pdf").exists()
    assert (dest_dir / "Acme.pdf").read_bytes() == b"already here"  # untouched


def test_archive_file_rejects_a_destination_outside_the_allowlist(tmp_path):
    src = tmp_path / "source.pdf"
    src.write_bytes(b"%PDF-fake")
    outside = tmp_path / "not_allowed"

    ok, message = automation.archive_file(str(src), str(outside), "x.pdf")
    assert ok is False
    assert "allowed archive locations" in message
    assert src.exists()  # nothing should have moved


def test_default_webhook_and_email_read_from_environment(monkeypatch):
    monkeypatch.setenv("NURU_WEBHOOK_URL", "https://hooks.example.com/x")
    monkeypatch.setenv("NURU_DEFAULT_EMAIL_TO", "ap@company.com")
    assert automation.default_webhook_url() == "https://hooks.example.com/x"
    assert automation.default_email_to() == "ap@company.com"


def test_allowed_archive_dirs_includes_extra_roots_and_stays_deduplicated(tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    monkeypatch.setenv("NURU_ARCHIVE_ALLOWED_ROOTS", os.pathsep.join([str(extra), str(extra)]))
    dirs = automation.allowed_archive_dirs()
    assert automation.DEFAULT_ARCHIVE_DIR in dirs
    assert str(extra) in dirs
    assert dirs.count(str(extra)) == 1
