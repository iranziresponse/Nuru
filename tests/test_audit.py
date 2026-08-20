import json

import audit


def test_record_writes_a_json_line_and_returns_the_entry(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.log"
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", str(log_path))

    entry = audit.record("scanned", ip="127.0.0.1", filename="a.pdf", type="Invoice")
    assert entry["event"] == "scanned"
    assert entry["ip"] == "127.0.0.1"
    assert "timestamp" in entry

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == entry


def test_record_never_captures_field_values():
    """The whole point of this module: it logs what happened, not the
    financial data that was in the document."""
    entry = audit.record("automated", filename="invoice.pdf", action="email")
    assert "fields" not in entry
    assert "value" not in entry


def test_read_recent_returns_most_recent_first_and_respects_limit(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.log"
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", str(log_path))

    for i in range(5):
        audit.record("scanned", filename=f"doc{i}.pdf")

    recent = audit.read_recent(limit=3)
    assert len(recent) == 3
    assert [e["filename"] for e in recent] == ["doc4.pdf", "doc3.pdf", "doc2.pdf"]


def test_read_recent_with_no_log_file_returns_empty_list(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", str(tmp_path / "does_not_exist.log"))
    assert audit.read_recent() == []


def test_read_recent_skips_corrupt_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.log"
    log_path.write_text('{"event": "scanned", "filename": "ok.pdf"}\nnot json\n', encoding="utf-8")
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", str(log_path))

    recent = audit.read_recent()
    assert len(recent) == 1
    assert recent[0]["filename"] == "ok.pdf"


def test_record_failure_to_write_does_not_raise(monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", "Z:\\this\\path\\cannot\\possibly\\exist\\audit.log")
    entry = audit.record("scanned", filename="a.pdf")  # must not raise
    assert entry["event"] == "scanned"


def test_summarize_counts_scanned_and_automated_events_by_action(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.log"
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", str(log_path))

    audit.record("scanned", filename="a.pdf")
    audit.record("scanned", filename="b.pdf")
    audit.record("automated", filename="a.pdf", action="webhook", ok=True)
    audit.record("automated", filename="b.pdf", action="webhook", ok=True)
    audit.record("automated", filename="a.pdf", action="ledger", ok=True)

    summary = audit.summarize()
    assert summary["total_scanned"] == 2
    assert summary["total_automated"] == 3
    assert summary["by_action"] == {"webhook": 2, "ledger": 1}
    assert summary["first_entry_at"] is not None
    assert summary["last_entry_at"] is not None


def test_summarize_returns_zeroed_result_when_log_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", str(tmp_path / "does_not_exist.log"))
    summary = audit.summarize()
    assert summary == {"total_scanned": 0, "total_automated": 0, "by_action": {},
                        "first_entry_at": None, "last_entry_at": None}


def test_summarize_skips_corrupt_lines(tmp_path, monkeypatch):
    log_path = tmp_path / "audit.log"
    log_path.write_text('{"event": "scanned", "filename": "ok.pdf"}\nnot json\n', encoding="utf-8")
    monkeypatch.setattr(audit, "AUDIT_LOG_PATH", str(log_path))

    summary = audit.summarize()
    assert summary["total_scanned"] == 1
