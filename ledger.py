"""Nuru's own persistent bookkeeping record.

Every other place an approved document can go (webhook, email, archive,
Excel download) is a one-shot handoff: once it leaves, Nuru keeps no
memory of it beyond the metadata-only audit trail. The ledger is the one
deliberately persistent store: an approved document's fields land in a
local SQLite database and stay there until someone explicitly deletes
them, so a scanned invoice becomes a real, searchable bookkeeping record
instead of a one-time payload.

Every mutating function returns (ok: bool, message: str) instead of
raising, matching automation.py's contract, so the caller can show the
result inline rather than crashing the review flow. Standard library
only (sqlite3), no new dependency.

Schema carries a little headroom for a possible future Odoo/ERPNext
export connector, without building any of that integration now:
  counterparty / party_type -> Odoo account.move.partner_id /
                                ERPNext "Purchase Invoice".supplier
  currency                  -> Odoo currency_id / ERPNext currency
  tax_amount                -> Odoo amount_tax / ERPNext total_taxes_and_charges
  document_number           -> Odoo ref / ERPNext bill_no
  account_code              -> Odoo account.account.code / ERPNext Account
document_number and account_code are reserved columns only: nothing in
Nuru populates them today, and there is no V1 UI for them.
"""

import datetime
import json
import os
import re
import sqlite3

import errors

_logger = errors.get_logger()

_LEDGER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nuru_data")
LEDGER_DB_PATH = os.path.join(_LEDGER_DIR, "ledger.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at         TEXT NOT NULL,
    document_date      TEXT,
    document_number    TEXT,
    original_filename  TEXT NOT NULL,
    doc_type           TEXT,
    counterparty       TEXT,
    party_type         TEXT,
    amount             REAL,
    amount_raw         TEXT,
    tax_amount         REAL,
    tax_amount_raw     TEXT,
    currency           TEXT DEFAULT 'USD',
    category           TEXT,
    account_code       TEXT,
    fields_json        TEXT NOT NULL,
    source_token       TEXT
);
CREATE INDEX IF NOT EXISTS idx_entries_document_date ON entries(document_date);
CREATE INDEX IF NOT EXISTS idx_entries_category ON entries(category);
CREATE INDEX IF NOT EXISTS idx_entries_doc_type ON entries(doc_type);
"""

# Label sets a saved field might carry, matched the same way
# automation.build_archive_filename already matches known labels.
_COUNTERPARTY_LABELS = ("Supplier Name", "Merchant", "Account Name")
_AMOUNT_LABELS = ("Total Amount Due", "Amount Paid", "Closing Balance")
_TAX_LABELS = ("Tax Value (VAT/GST)", "Sales Tax", "Tax")
_DATE_LABELS = ("Transaction Date", "Purchase Date", "Statement Period")

_PARTY_TYPE_BY_DOC_TYPE = {
    "Invoice": "supplier",
    "Receipt": "supplier",
    "Statement": "account_holder",
}

_AMOUNT_JUNK = re.compile(r"[^0-9.\-]")


def get_connection():
    """One connection per call; callers are expected to use it as a
    context manager or close it. Ensures the schema exists first, since
    there's no separate migration step to run before this is called."""
    os.makedirs(os.path.dirname(LEDGER_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(LEDGER_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _parse_amount(raw):
    """"$1,234.56" -> 1234.56, "(200.00)" -> -200.0 (parenthesized is a
    common statement convention for a credit/negative figure), "n/a" or
    "" -> None. Never raises."""
    if not raw:
        return None
    text = raw.strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    cleaned = _AMOUNT_JUNK.sub("", text)
    if not cleaned or cleaned in ("-", "."):
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def _first_value(rows, labels):
    return next((r["value"] for r in rows if r["label"] in labels and r["value"]), None)


def save_entry(rows, record, token, category=None):
    """rows is the same [{label, value}] list every automation.* function
    receives from webapp.py's _parse_rows(). record is the review record
    (needs "original_filename" and "type"). category is the user's choice
    from the Automate step's ledger param, mirroring how webhook/email/
    archive each take their own param (URL/address/folder). fields_json
    always keeps the full approved set, so a renamed/custom field is
    never lost even when it doesn't match a known label for the summary
    columns."""
    counterparty = _first_value(rows, _COUNTERPARTY_LABELS)
    amount_raw = _first_value(rows, _AMOUNT_LABELS)
    tax_raw = _first_value(rows, _TAX_LABELS)
    document_date = _first_value(rows, _DATE_LABELS)
    doc_type = record.get("type")

    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO entries
                   (created_at, document_date, original_filename, doc_type,
                    counterparty, party_type, amount, amount_raw,
                    tax_amount, tax_amount_raw, category, fields_json, source_token)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    document_date,
                    record.get("original_filename") or "Unknown document",
                    doc_type,
                    counterparty,
                    _PARTY_TYPE_BY_DOC_TYPE.get(doc_type),
                    _parse_amount(amount_raw),
                    amount_raw,
                    _parse_amount(tax_raw),
                    tax_raw,
                    (category or "").strip() or None,
                    json.dumps(rows),
                    token,
                ),
            )
    except sqlite3.Error as exc:
        errors.report_exception(exc, action="ledger_save_entry", token=token)
        return False, "Couldn't save this to the ledger. Please try again."

    return True, "Saved to the ledger."


def _where_clause(q, doc_type, category, date_from, date_to):
    clauses = []
    params = []
    if q:
        like = f"%{q}%"
        clauses.append(
            "(original_filename LIKE ? OR counterparty LIKE ? OR category LIKE ? OR fields_json LIKE ?)"
        )
        params += [like, like, like, like]
    if doc_type:
        clauses.append("doc_type = ?")
        params.append(doc_type)
    if category:
        clauses.append("category = ?")
        params.append(category)
    if date_from:
        clauses.append("document_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("document_date <= ?")
        params.append(date_to)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


def list_entries(q=None, doc_type=None, category=None, date_from=None, date_to=None,
                  limit=100, offset=0):
    """Returns (rows, has_more). Sorted by document_date DESC (NULLs
    last, since a row with no recognized date shouldn't hide newer,
    dated ones above it), then created_at DESC as a tiebreaker."""
    where, params = _where_clause(q, doc_type, category, date_from, date_to)
    sql = f"""
        SELECT * FROM entries {where}
        ORDER BY (document_date IS NULL) ASC, document_date DESC, created_at DESC
        LIMIT ? OFFSET ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, params + [limit + 1, offset]).fetchall()
    has_more = len(rows) > limit
    return [dict(r) for r in rows[:limit]], has_more


def get_totals(q=None, doc_type=None, category=None, date_from=None, date_to=None):
    """Aggregates over the whole matching set, not just the current page."""
    where, params = _where_clause(q, doc_type, category, date_from, date_to)
    with get_connection() as conn:
        rows = conn.execute(f"SELECT doc_type, amount FROM entries {where}", params).fetchall()

    count = len(rows)
    recognized = [r["amount"] for r in rows if r["amount"] is not None]
    sum_amount = sum(recognized) if recognized else None
    unrecognized_amount_count = count - len(recognized)

    by_type = {}
    for r in rows:
        key = r["doc_type"] or "Unknown"
        bucket = by_type.setdefault(key, {"count": 0, "sum": 0.0})
        bucket["count"] += 1
        if r["amount"] is not None:
            bucket["sum"] += r["amount"]

    return {
        "count": count,
        "sum_amount": sum_amount,
        "unrecognized_amount_count": unrecognized_amount_count,
        "by_type": by_type,
    }


def distinct_categories():
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM entries WHERE category IS NOT NULL AND category != '' "
            "ORDER BY category"
        ).fetchall()
    return [r["category"] for r in rows]


def delete_entry(entry_id):
    try:
        with get_connection() as conn:
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    except sqlite3.Error as exc:
        errors.report_exception(exc, action="ledger_delete_entry", entry_id=entry_id)
        return False, "Couldn't delete that entry. Please try again."
    return True, "Deleted."
