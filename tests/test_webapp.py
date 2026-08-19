"""Flask route tests. Critically isolated from the real app's on-disk
cache/state file (webapp._RESULTS_DIR / _STATE_PATH) via monkeypatch, since
importing webapp.py runs its module-level state exactly once per test
session and a real Nuru instance may be running against the same files."""

import io
import json
import os

import pytest

pytest.importorskip("reportlab.pdfgen.canvas")
from reportlab.pdfgen import canvas  # noqa: E402

import webapp  # noqa: E402


def _make_pdf_bytes(lines):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    buf.seek(0)
    return buf.read()


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Point the app at a throwaway cache dir and a fresh in-memory
    _PENDING for every test, so tests never touch a real running
    instance's cache or persisted state."""
    cache_dir = tmp_path / "nuru_cache"
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setattr(webapp, "_RESULTS_DIR", str(cache_dir))
    monkeypatch.setattr(webapp, "_STATE_PATH", str(cache_dir / "pending_state.json"))
    monkeypatch.setattr(webapp, "_PENDING", {})
    monkeypatch.delenv("NURU_ACCESS_PASSWORD", raising=False)
    yield


@pytest.fixture
def client():
    """CSRF and rate limiting are exercised by their own dedicated tests
    below; disabling them here is the standard way to test route logic
    without every other test having to carry a valid CSRF token or worry
    about tripping a shared rate-limit bucket across the whole test run."""
    webapp.app.config["TESTING"] = True
    webapp.app.config["WTF_CSRF_ENABLED"] = False
    webapp.app.config["RATELIMIT_ENABLED"] = False
    return webapp.app.test_client()


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Nuru" in resp.data


def test_scan_with_no_files_shows_friendly_error(client):
    resp = client.post("/scan", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"choose at least one PDF" in resp.data


def test_review_unknown_token_returns_expired_page(client):
    resp = client.get("/review/does-not-exist")
    assert resp.status_code == 404
    assert b"gone" in resp.data.lower()


def test_full_scan_review_automate_discard_flow(client):
    pdf_bytes = _make_pdf_bytes([
        "Quantum Electronics", "Invoice Date: 2026-04-02",
        "Total Due: $500.00", "Tax (8%): $40.00",
    ])
    resp = client.post(
        "/scan",
        data={"pdfs": (io.BytesIO(pdf_bytes), "invoice.pdf")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"invoice.pdf" in resp.data
    assert b"Invoice" in resp.data  # type badge

    assert len(webapp._PENDING) == 1
    token = next(iter(webapp._PENDING))
    record = webapp._PENDING[token]
    assert record["pdf_path"] is not None
    assert os.path.exists(record["pdf_path"])

    automate_resp = client.post(
        f"/automate/{token}",
        json={"fields": [], "action": "discard", "param": ""},
    )
    assert automate_resp.status_code == 200
    body = automate_resp.get_json()
    assert body["ok"] is True
    assert webapp._PENDING[token]["done"] is True
    # Discarding must purge the cached source file.
    assert not os.path.exists(record["pdf_path"])


def test_automate_unknown_action_returns_400(client):
    pdf_bytes = _make_pdf_bytes(["Acme", "Total Due: $10.00"])
    client.post("/scan", data={"pdfs": (io.BytesIO(pdf_bytes), "a.pdf")},
                content_type="multipart/form-data")
    token = next(iter(webapp._PENDING))
    resp = client.post(f"/automate/{token}", json={"fields": [], "action": "not-a-real-action"})
    assert resp.status_code == 400


def test_automate_unknown_token_returns_404(client):
    resp = client.post("/automate/does-not-exist", json={"fields": [], "action": "discard"})
    assert resp.status_code == 404
    assert resp.get_json()["ok"] is False


def test_corrupt_pdf_in_batch_does_not_break_the_scan(client):
    good = _make_pdf_bytes(["Acme", "Total Due: $10.00"])
    resp = client.post(
        "/scan",
        data={"pdfs": [
            (io.BytesIO(good), "good.pdf"),
            (io.BytesIO(b"not a real pdf"), "bad.pdf"),
        ]},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert len(webapp._PENDING) == 2
    errors = [r["error"] for r in webapp._PENDING.values()]
    assert errors.count(True) == 1
    assert errors.count(False) == 1


def test_access_password_gate(client, monkeypatch):
    monkeypatch.setenv("NURU_ACCESS_PASSWORD", "letmein")
    resp = client.get("/")
    assert resp.status_code == 401

    import base64
    creds = base64.b64encode(b"user:letmein").decode()
    resp2 = client.get("/", headers={"Authorization": f"Basic {creds}"})
    assert resp2.status_code == 200


def test_security_headers_present_on_every_response(client):
    resp = client.get("/")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in resp.headers["Content-Security-Policy"]


def test_csrf_protection_blocks_a_tokenless_post():
    """Uses its own client (CSRF left ON) rather than the shared `client`
    fixture, which disables CSRF for route-logic tests elsewhere."""
    webapp.app.config["TESTING"] = True
    webapp.app.config["WTF_CSRF_ENABLED"] = True
    webapp.app.config["RATELIMIT_ENABLED"] = False
    raw_client = webapp.app.test_client()
    try:
        resp = raw_client.post(
            "/automate/some-token",
            data=json.dumps({"fields": [], "action": "discard"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
    finally:
        webapp.app.config["WTF_CSRF_ENABLED"] = False


def test_rate_limit_blocks_requests_past_the_cap():
    """Uses its own client (rate limiting left ON)."""
    webapp.app.config["TESTING"] = True
    webapp.app.config["WTF_CSRF_ENABLED"] = False
    webapp.app.config["RATELIMIT_ENABLED"] = True
    raw_client = webapp.app.test_client()
    try:
        statuses = [raw_client.get("/").status_code for _ in range(65)]
        assert 429 in statuses
    finally:
        webapp.app.config["RATELIMIT_ENABLED"] = False
