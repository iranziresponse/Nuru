import json
import os

import pytest

import ledger


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_DB_PATH", str(tmp_path / "ledger.db"))
    yield


INVOICE_ROWS = [
    {"label": "Supplier Name", "value": "Acme Co"},
    {"label": "Transaction Date", "value": "2026-04-02"},
    {"label": "Total Amount Due", "value": "$1,339.20"},
    {"label": "Tax Value (VAT/GST)", "value": "$99.20"},
]
INVOICE_RECORD = {"original_filename": "invoice.pdf", "type": "Invoice"}


# ---- _parse_amount ---------------------------------------------------------

def test_parse_amount_handles_currency_symbols_and_commas():
    assert ledger._parse_amount("$1,234.56") == 1234.56


def test_parse_amount_treats_parens_as_negative():
    assert ledger._parse_amount("(200.00)") == -200.0


def test_parse_amount_returns_none_for_garbage():
    assert ledger._parse_amount("n/a") is None
    assert ledger._parse_amount("") is None
    assert ledger._parse_amount(None) is None


# ---- save_entry -------------------------------------------------------------

def test_save_entry_writes_a_row_and_denormalizes_known_labels():
    ok, message = ledger.save_entry(INVOICE_ROWS, INVOICE_RECORD, "tok-1", category="Travel")
    assert ok is True

    entries, has_more = ledger.list_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["counterparty"] == "Acme Co"
    assert entry["amount"] == 1339.20
    assert entry["amount_raw"] == "$1,339.20"
    assert entry["tax_amount"] == 99.20
    assert entry["document_date"] == "2026-04-02"
    assert entry["doc_type"] == "Invoice"
    assert entry["party_type"] == "supplier"
    assert entry["category"] == "Travel"
    assert entry["source_token"] == "tok-1"
    assert entry["currency"] == "USD"
    assert has_more is False


def test_save_entry_keeps_full_fields_json_even_when_labels_are_unrecognized():
    rows = [{"label": "My Custom Field", "value": "Whatever"}]
    ok, message = ledger.save_entry(rows, INVOICE_RECORD, "tok-2")
    assert ok is True

    entries, _ = ledger.list_entries()
    entry = entries[0]
    assert entry["counterparty"] is None
    assert entry["amount"] is None
    assert entry["document_date"] is None
    assert json.loads(entry["fields_json"]) == rows


def test_save_entry_failure_returns_ok_false_not_an_exception(tmp_path, monkeypatch):
    unwritable = tmp_path / "does" / "not" / "exist" / "ledger.db"
    monkeypatch.setattr(ledger, "get_connection", lambda: (_ for _ in ()).throw(ledger.sqlite3.Error("boom")))
    ok, message = ledger.save_entry(INVOICE_ROWS, INVOICE_RECORD, "tok-3")
    assert ok is False
    assert "try again" in message.lower()


# ---- list_entries / filters -------------------------------------------------

def _seed():
    ledger.save_entry(
        [{"label": "Supplier Name", "value": "Acme Co"},
         {"label": "Transaction Date", "value": "2026-01-01"},
         {"label": "Total Amount Due", "value": "$100.00"}],
        {"original_filename": "a.pdf", "type": "Invoice"}, "t-a", category="Travel",
    )
    ledger.save_entry(
        [{"label": "Merchant", "value": "Copperline Hardware"},
         {"label": "Purchase Date", "value": "2026-02-01"},
         {"label": "Amount Paid", "value": "$47.88"}],
        {"original_filename": "b.pdf", "type": "Receipt"}, "t-b", category="Office Supplies",
    )
    ledger.save_entry(
        [{"label": "Account Name", "value": "Jordan Ferreira"},
         {"label": "Statement Period", "value": "2026-03-01"},
         {"label": "Closing Balance", "value": "$9,214.60"}],
        {"original_filename": "c.pdf", "type": "Statement"}, "t-c",
    )


def test_list_entries_filters_by_search_type_category_and_date_range():
    _seed()

    by_search, _ = ledger.list_entries(q="Copperline")
    assert [e["original_filename"] for e in by_search] == ["b.pdf"]

    by_type, _ = ledger.list_entries(doc_type="Statement")
    assert [e["original_filename"] for e in by_type] == ["c.pdf"]

    by_category, _ = ledger.list_entries(category="Travel")
    assert [e["original_filename"] for e in by_category] == ["a.pdf"]

    by_range, _ = ledger.list_entries(date_from="2026-02-01", date_to="2026-02-28")
    assert [e["original_filename"] for e in by_range] == ["b.pdf"]


def test_list_entries_sorts_newest_document_date_first():
    _seed()
    entries, _ = ledger.list_entries()
    assert [e["original_filename"] for e in entries] == ["c.pdf", "b.pdf", "a.pdf"]


def test_list_entries_has_more_flag_and_offset_pagination():
    _seed()
    page1, has_more1 = ledger.list_entries(limit=2, offset=0)
    assert len(page1) == 2
    assert has_more1 is True

    page2, has_more2 = ledger.list_entries(limit=2, offset=2)
    assert len(page2) == 1
    assert has_more2 is False


# ---- get_totals --------------------------------------------------------------

def test_get_totals_sums_only_recognized_amounts_and_breaks_down_by_type():
    _seed()
    ledger.save_entry([{"label": "My Field", "value": "x"}],
                       {"original_filename": "d.pdf", "type": "Invoice"}, "t-d")

    totals = ledger.get_totals()
    assert totals["count"] == 4
    assert totals["sum_amount"] == pytest.approx(100.00 + 47.88 + 9214.60)
    assert totals["unrecognized_amount_count"] == 1
    assert totals["by_type"]["Invoice"]["count"] == 2
    assert totals["by_type"]["Receipt"]["count"] == 1
    assert totals["by_type"]["Statement"]["sum"] == pytest.approx(9214.60)


def test_get_totals_respects_filters():
    _seed()
    totals = ledger.get_totals(doc_type="Receipt")
    assert totals["count"] == 1
    assert totals["sum_amount"] == pytest.approx(47.88)


# ---- distinct_categories / delete_entry --------------------------------------

def test_distinct_categories_excludes_empty_and_sorts():
    _seed()
    assert ledger.distinct_categories() == ["Office Supplies", "Travel"]


def test_delete_entry_removes_the_row():
    _seed()
    entries, _ = ledger.list_entries()
    target_id = entries[0]["id"]

    ok, message = ledger.delete_entry(target_id)
    assert ok is True

    remaining, _ = ledger.list_entries()
    assert target_id not in [e["id"] for e in remaining]
    assert len(remaining) == 2


def test_delete_entry_failure_returns_ok_false_not_an_exception(monkeypatch):
    monkeypatch.setattr(ledger, "get_connection", lambda: (_ for _ in ()).throw(ledger.sqlite3.Error("boom")))
    ok, message = ledger.delete_entry(1)
    assert ok is False
