"""Nuru: a local browser page for reading invoices, receipts, and financial
statements as a human-in-the-loop pipeline.

Three phases, each its own screen: Scan (upload) -> Review (an editable
grid of whatever fields this document's kind actually has, with room to
rename, remove, or add fields) -> Automate (send the approved data to a
webhook, email it, or archive the source file under a clean name). The
source PDF stays cached only as long as a document is mid-review, and is
permanently deleted the moment its chosen automation action finishes.

Usage:
    python webapp.py
    -> open http://127.0.0.1:5000
"""

import datetime
import json
import os
import re
import secrets
import uuid
from urllib.parse import urlencode

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import CSRFProtect
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import audit
import automation
import errors
import ledger
from app import (
    DEFAULT_CONFIDENCE_THRESHOLD, VOCAB_PATH, WEIGHTS_PATH,
    process_invoice, write_excel,
)
from engine.model import TokenClassifier
from engine.tokenizer import Vocabulary

_logger = errors.get_logger()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB per request; blocks giant/DoS-y uploads

# In-memory storage is fine for a single-process local instance; a real
# multi-worker deployment would need a shared backend (e.g. Redis) so
# workers share one count instead of each enforcing its own.
limiter = Limiter(get_remote_address, app=app, default_limits=["60 per minute"], storage_uri="memory://")

# A stable local cache, not the OS temp dir: cached PDFs are deleted
# explicitly the moment their automation runs (see _purge_cached_pdf), so
# nothing here relies on the OS clearing temp files for privacy. Being
# stable means a server restart doesn't lose documents mid-review.
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nuru_cache")
os.makedirs(_RESULTS_DIR, exist_ok=True)
_STATE_PATH = os.path.join(_RESULTS_DIR, "pending_state.json")

_model = None
_vocab = None
_PENDING = {}  # token -> review record, see scan() for shape


def _save_state():
    try:
        with open(_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(_PENDING, f)
    except OSError as exc:
        _logger.warning(f"could not save pending state: {exc}")


def _load_state():
    global _PENDING
    if not os.path.exists(_STATE_PATH):
        return
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            _PENDING = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning(f"could not load pending state, starting fresh: {exc}")
        _PENDING = {}


_load_state()


def _get_or_create_secret_key():
    """Flask-WTF's CSRF tokens are signed with this. NURU_SECRET_KEY wins if
    set; otherwise a random key is generated once and persisted alongside
    the other local state, so a server restart doesn't invalidate the CSRF
    token already embedded in a page someone has open mid-review."""
    env_key = os.environ.get("NURU_SECRET_KEY")
    if env_key:
        return env_key
    key_path = os.path.join(_RESULTS_DIR, "secret_key")
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            existing = f.read().strip()
            if existing:
                return existing
    key = secrets.token_hex(32)
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(key)
    return key


app.config["SECRET_KEY"] = _get_or_create_secret_key()
csrf = CSRFProtect(app)
app.jinja_env.filters["from_json"] = json.loads

# A few common alternate phrasings per underlying entity, offered as a
# shortcut in the review grid's label dropdown. The label text field next
# to it is always freely editable regardless.
PRESET_LABELS = {
    "Vendor": ["Supplier Name", "Vendor", "Vendor Entity", "Company Name"],
    "Merchant": ["Merchant", "Store Name", "Retailer"],
    "Account": ["Account Name", "Account Holder", "Customer Name"],
    "Date": ["Transaction Date", "Purchase Date", "Invoice Date", "Date"],
    "Total": ["Total Amount Due", "Amount Paid", "Total"],
    "Tax": ["Tax Value (VAT/GST)", "Sales Tax", "Tax"],
    "PaymentMethod": ["Payment Method", "Payment Type"],
    "Period": ["Statement Period", "Billing Period", "Period"],
    "Balance": ["Closing Balance", "Balance Due", "Balance"],
}

# Convenience presets for the ledger's category field. Freetext always wins
# here too, same convention as PRESET_LABELS above.
LEDGER_CATEGORIES = [
    "Office Supplies", "Software & Subscriptions", "Travel",
    "Meals & Entertainment", "Utilities", "Professional Services",
    "Rent", "Insurance", "Taxes", "Client Reimbursable", "Other",
]


def get_model():
    global _model, _vocab
    if _model is None:
        if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(VOCAB_PATH):
            raise RuntimeError("Nuru hasn't been trained yet. Run `python train.py` first.")
        _vocab = Vocabulary.load(VOCAB_PATH)
        _model = TokenClassifier.load_weights(WEIGHTS_PATH)
    return _model, _vocab


def _purge_cached_pdf(pdf_path):
    if not pdf_path:
        return
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    except OSError as exc:
        _logger.warning(f"could not purge cached PDF {pdf_path}: {exc}")


_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _default_download_filename(record, token):
    """No extension: callers append .xlsx themselves, since this same base
    also seeds the editable filename field a user can override before the
    file is actually requested."""
    stem = os.path.splitext(record["original_filename"])[0] if record else "Invoice"
    return f"Nuru - {stem} ({token[:6]})"


def _sanitize_download_filename(name, fallback):
    """A name someone typed by hand needs the same treatment a filesystem
    or a browser would otherwise apply silently: strip characters that are
    illegal in a Windows filename, drop a redundant .xlsx they may have
    typed themselves so it isn't doubled, and fall back rather than ever
    hand a blank name to send_file."""
    name = re.sub(r"\.xlsx$", "", (name or "").strip(), flags=re.IGNORECASE)
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" .")
    name = name[:150]
    return name or fallback


def _next_pending_token(record):
    tokens = record["batch_tokens"]
    for t in tokens[record["batch_index"] + 1:]:
        other = _PENDING.get(t)
        if other and not other["done"]:
            return t
    return None


REVIEW_TTL_HOURS = float(os.environ.get("NURU_REVIEW_TTL_HOURS", "24"))


def _purge_stale_pending():
    """A document that's scanned but never reviewed to completion would
    otherwise sit in the cache indefinitely: the retention story ("gone
    the moment its automation completes") only holds if someone actually
    finishes that step. Anything left mid-review past the TTL gets its
    cached source file deleted and is marked done, same as if it had been
    explicitly discarded."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=REVIEW_TTL_HOURS)
    changed = False
    for token, record in _PENDING.items():
        if record["done"]:
            continue
        created_at = record.get("created_at")
        if not created_at:
            continue
        try:
            created = datetime.datetime.fromisoformat(created_at)
        except ValueError:
            continue
        if created < cutoff:
            _purge_cached_pdf(record.get("pdf_path"))
            record["pdf_path"] = None
            record["done"] = True
            audit.record("auto_purged", filename=record["original_filename"],
                          type=record["type"], reason="review left incomplete past TTL")
            changed = True
    if changed:
        _save_state()


# ---------------------------------------------------------------- styling --
BASE_STYLE = """
  :root {
    --bg: #f5f7fb; --card: #ffffff; --text: #16203d; --text-soft: #6b7590;
    --line: #e6e9f2; --navy: #16205c; --navy-dark: #0e1740;
    --cyan: #27c6ee; --cyan-tint: #e8f9fd;
    --accent: #16205c; --accent-dark: #0e1740; --accent-tint: #e8f9fd;
    --ok: #2f6f52; --ok-tint: #e9f4ee;
    --warn: #a8621a; --warn-tint: #fbedd8;
    --error: #a8402f; --error-tint: #faeae6;
    --shadow: 0 20px 45px -20px rgba(22, 32, 92, 0.22);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(1100px 500px at 50% -10%, #eafaff 0%, var(--bg) 55%);
    color: var(--text);
    font-family: 'Inter', -apple-system, 'Segoe UI', system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .page { max-width: 640px; margin: 0 auto; padding: 56px 24px 40px; min-height: 100vh; }
  header { display: flex; align-items: center; gap: 18px; margin-bottom: 36px; }
  .logo-link { display: flex; align-items: center; gap: 10px; text-decoration: none; }
  .logo-mark { flex-shrink: 0; }
  .logo-word { font-weight: 800; font-size: 1.2rem; letter-spacing: -0.01em; color: var(--navy); }
  .nav-links { margin-left: auto; display: flex; gap: 20px; }
  .nav-link { font-size: 0.85rem; font-weight: 600; color: var(--text-soft); text-decoration: none;
              padding: 4px 1px; border-bottom: 2px solid transparent; }
  .nav-link:hover { color: var(--navy); }
  .nav-link.active { color: var(--navy); border-bottom-color: var(--cyan); }
  h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 8px; }
  .tagline { color: var(--text-soft); font-size: 1rem; line-height: 1.55; margin: 0 0 32px; }

  .stepper { display: flex; align-items: center; gap: 8px; margin-bottom: 28px; }
  .step { font-size: 0.82rem; font-weight: 600; color: var(--text-soft); padding: 5px 12px;
          border-radius: 999px; background: var(--card); border: 1px solid var(--line); }
  .step.active { color: #fff; background: var(--navy); border-color: var(--navy); }
  .step-arrow { color: var(--text-soft); font-size: 0.8rem; }

  .card { background: var(--card); border-radius: 18px; padding: 26px; box-shadow: var(--shadow);
          border: 1px solid var(--line); }

  .dropzone { display: flex; flex-direction: column; align-items: center; gap: 10px; text-align: center;
              padding: 44px 24px; border: 1.5px dashed #cdd3e6; border-radius: 18px; background: var(--card);
              cursor: pointer; transition: border-color 0.15s ease, background 0.15s ease, transform 0.1s ease; }
  .dropzone:hover, .dropzone.dragging { border-color: var(--accent); background: var(--accent-tint); }
  .dropzone.dragging { transform: scale(1.01); }
  .dz-title { font-weight: 600; font-size: 1.02rem; }
  .dz-sub { color: var(--text-soft); font-size: 0.9rem; }
  .dz-filename { font-weight: 600; color: var(--accent-dark); font-size: 0.95rem; }

  .btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; width: 100%;
         padding: 14px; border: none; border-radius: 12px; font-family: inherit; font-size: 0.98rem;
         font-weight: 600; cursor: pointer; transition: background 0.15s ease, opacity 0.15s ease; }
  .btn-primary { background: var(--navy); color: #fff; }
  .btn-primary:hover { background: var(--navy-dark); }
  .btn-primary:disabled { opacity: 0.6; cursor: wait; }
  .btn-ghost { background: transparent; color: var(--text-soft); border: 1px solid var(--line); }
  .btn-ghost:hover { background: #f4f5fa; }
  .btn-small { width: auto; padding: 8px 14px; font-size: 0.85rem; }

  .notice { margin-top: 18px; padding: 14px 16px; border-radius: 12px; font-size: 0.95rem; line-height: 1.5; }
  .notice-error { background: var(--error-tint); color: var(--error); }
  .notice-ok { background: var(--ok-tint); color: var(--ok); }

  footer { margin-top: 40px; text-align: center; }
  footer p { color: var(--text-soft); font-size: 0.82rem; line-height: 1.6; margin: 0; }
"""

REVIEW_STYLE = """
  .doc-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 22px; }
  .doc-name { font-weight: 600; font-size: 0.98rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .badge { flex-shrink: 0; font-size: 0.78rem; font-weight: 600; padding: 5px 11px; border-radius: 999px; }
  .badge-type { background: var(--cyan-tint); color: var(--navy); }
  .badge-warn { background: var(--warn-tint); color: var(--warn); }

  .field-row { display: grid; grid-template-columns: 1fr 1fr auto; gap: 8px; align-items: center;
               padding: 10px 0; border-top: 1px solid var(--line); }
  .field-row:first-child { border-top: none; }
  .field-row.flagged { background: var(--warn-tint); margin: 0 -12px; padding: 10px 12px; border-radius: 10px; border-top: none; }
  .field-label-group { display: flex; flex-direction: column; gap: 4px; }
  .field-row select, .field-row input[type=text] {
    width: 100%; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px;
    font-family: inherit; font-size: 0.9rem; color: var(--text); background: #fff;
  }
  .field-row select { font-size: 0.78rem; color: var(--text-soft); padding: 5px 8px; }
  .field-row input:focus, .field-row select:focus { outline: 2px solid var(--cyan); outline-offset: 1px; }
  .remove-btn { background: none; border: none; color: var(--text-soft); cursor: pointer; font-size: 1.1rem;
                width: 28px; height: 28px; border-radius: 8px; }
  .remove-btn:hover { background: var(--error-tint); color: var(--error); }

  .add-field-btn { margin-top: 14px; background: none; border: 1px dashed #cdd3e6; color: var(--text-soft);
                    border-radius: 10px; padding: 10px; width: 100%; font-family: inherit; font-size: 0.88rem;
                    cursor: pointer; }
  .add-field-btn:hover { border-color: var(--accent); color: var(--accent); }

  .actions-row { display: flex; gap: 10px; margin-top: 22px; }

  .action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 18px 0; }
  .action-card { border: 1.5px solid var(--line); border-radius: 12px; padding: 14px; cursor: pointer;
                  text-align: left; background: #fff; }
  .action-card.selected { border-color: var(--navy); background: var(--cyan-tint); }
  .action-card .a-title { font-weight: 600; font-size: 0.9rem; display: block; margin-bottom: 2px; }
  .action-card .a-sub { font-size: 0.78rem; color: var(--text-soft); }
  .action-param { margin-top: 14px; }
  .action-param input[type=text], .action-param select { width: 100%; padding: 10px 12px;
    border: 1px solid var(--line); border-radius: 10px; font-family: inherit; font-size: 0.92rem;
    background: #fff; color: var(--text); }
  .preview-box { margin: 10px 0 0; padding: 14px 16px; background: var(--cyan-tint);
    border: 1px solid var(--line); border-radius: 10px; font-family: monospace; font-size: 0.82rem;
    white-space: pre-wrap; word-break: break-word; color: var(--text); max-height: 260px; overflow-y: auto; }

  .result-icon { font-size: 1.6rem; }
  .next-link { display: block; text-align: center; margin-top: 16px; color: var(--navy); font-weight: 600;
               text-decoration: none; font-size: 0.92rem; }
"""

LOGO_SVG = """<svg class="logo-mark" width="30" height="30" viewBox="0 0 100 100">
      <path d="M50 4 L34 34 L4 50 L34 66 L50 96 Z" fill="#16205c"/>
      <path d="M50 4 L66 34 L96 50 L66 66 L50 96 Z" fill="#27c6ee"/>
    </svg>"""

FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
           "%3Cpath d='M50 4 L34 34 L4 50 L34 66 L50 96 Z' fill='%2316205c'/%3E"
           "%3Cpath d='M50 4 L66 34 L96 50 L66 66 L50 96 Z' fill='%2327c6ee'/%3E%3C/svg%3E")

# Shared across every page so there's always a way back to the upload
# screen and the audit trail, not just the browser's back button.
HEADER_NAV = """
  <header>
    <a class="logo-link" href="/">""" + LOGO_SVG + """<span class="logo-word">Nuru</span></a>
    <nav class="nav-links">
      <a class="nav-link{{ ' active' if active == 'upload' else '' }}" href="/">New scan</a>
      <a class="nav-link{{ ' active' if active == 'ledger' else '' }}" href="/ledger">Ledger</a>
      <a class="nav-link{{ ' active' if active == 'audit' else '' }}" href="/audit">Audit trail</a>
    </nav>
  </header>
"""

HEAD = f"""
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""

UPLOAD_PAGE = """
<!doctype html>
<html lang="en">
<head>
""" + HEAD + """
<title>Nuru | Read your documents</title>
<style>""" + BASE_STYLE + """</style>
</head>
<body>
<div class="page">
""" + HEADER_NAV + """
  <main>
    <h1>Give Nuru your documents.</h1>
    <p class="tagline">Drop in invoices, receipts, or statements. Nuru reads each one, then lets you review and approve every field before anything leaves the app.</p>

    <form method="post" action="/scan" enctype="multipart/form-data" id="upload-form">
      <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
      <label class="dropzone" id="dropzone" for="pdf-input">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none">
          <path d="M12 3v12M12 3l-4 4M12 3l4 4" stroke="#16205c" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M4 16v2.5A2.5 2.5 0 0 0 6.5 21h11a2.5 2.5 0 0 0 2.5-2.5V16" stroke="#16205c" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
        <span class="dz-title" id="dz-label">Drop your documents here</span>
        <span class="dz-sub" id="dz-sub">or click to browse, or paste with Ctrl+V. One or more PDF files</span>
        <input type="file" id="pdf-input" name="pdfs" accept="application/pdf" multiple required hidden>
      </label>
      <div style="margin-top: 18px;">
        <button type="submit" class="btn btn-primary" id="submit-btn">Scan documents</button>
      </div>
    </form>

    {% if error %}<div class="notice notice-error">{{ error }}</div>{% endif %}
  </main>

  <footer><p>Each file is read only for as long as it takes you to review and approve it.<br>Nothing is kept once you've sent it on.</p></footer>
</div>

<script>
  const dropzone = document.getElementById('dropzone');
  const input = document.getElementById('pdf-input');
  const dzLabel = document.getElementById('dz-label');
  const dzSub = document.getElementById('dz-sub');
  const form = document.getElementById('upload-form');
  const submitBtn = document.getElementById('submit-btn');

  function showFiles(files) {
    if (files.length === 1) { dzLabel.textContent = files[0].name; }
    else { dzLabel.textContent = files.length + ' files selected'; }
    dzLabel.classList.add('dz-filename');
    dzSub.textContent = 'Ready. Click "Scan documents" below';
  }
  input.addEventListener('change', () => { if (input.files.length) showFiles(input.files); });
  ['dragenter', 'dragover'].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(evt => dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('dragging'); }));
  dropzone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length) { input.files = files; showFiles(files); }
  });

  document.addEventListener('paste', e => {
    const clipboard = e.clipboardData;
    if (!clipboard || !clipboard.files || !clipboard.files.length) return;
    const pdfFiles = Array.from(clipboard.items)
      .filter(item => item.kind === 'file')
      .map(item => item.getAsFile())
      .filter(file => file && (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')));
    if (!pdfFiles.length) return;
    e.preventDefault();
    const transfer = new DataTransfer();
    pdfFiles.forEach(file => transfer.items.add(file));
    input.files = transfer.files;
    showFiles(input.files);
  });

  form.addEventListener('submit', () => { submitBtn.disabled = true; submitBtn.textContent = 'Reading your documents…'; });
</script>
</body>
</html>
"""

EXPIRED_PAGE = """
<!doctype html>
<html lang="en">
<head>""" + HEAD + """<title>Nuru</title><style>""" + BASE_STYLE + """</style></head>
<body>
<div class="page">
""" + HEADER_NAV + """
  <main>
    <h1>This review session is gone.</h1>
    <p class="tagline">It may have already been completed, or it's simply expired. Start again with a fresh upload.</p>
    <a class="btn btn-primary" style="text-decoration:none; display:inline-flex;" href="/">Upload documents</a>
  </main>
</div>
</body>
</html>
"""

AUDIT_STYLE = """
  .audit-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .audit-table th { text-align: left; padding: 8px 10px; color: var(--text-soft); font-weight: 600;
                     border-bottom: 1px solid var(--line); white-space: nowrap; }
  .audit-table td { padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
  .audit-table tr:last-child td { border-bottom: none; }
  .audit-time { font-family: var(--font-mono, inherit); color: var(--text-soft); white-space: nowrap; font-size: 0.8rem; }
  .audit-ok { color: var(--ok); font-weight: 600; }
  .audit-fail { color: var(--error); font-weight: 600; }
  .audit-empty { color: var(--text-soft); padding: 24px 0; text-align: center; }
  .table-scroll { overflow-x: auto; }
"""

AUDIT_PAGE = """
<!doctype html>
<html lang="en">
<head>""" + HEAD + """<title>Nuru | Audit trail</title><style>""" + BASE_STYLE + AUDIT_STYLE + """</style></head>
<body>
<div class="page">
""" + HEADER_NAV + """
  <main>
    <h1>Audit trail</h1>
    <p class="tagline">Every document scanned and every automation run, most recent first. Records metadata only: who, when, where, and whether it succeeded, never the extracted field values themselves.</p>

    <div class="card">
      {% if entries %}
      <div class="table-scroll">
        <table class="audit-table">
          <tr><th>Time (UTC)</th><th>Event</th><th>Document</th><th>Type</th><th>Detail</th><th>Outcome</th><th>From</th></tr>
          {% for e in entries %}
          <tr>
            <td class="audit-time">{{ e.timestamp }}</td>
            <td>{{ e.event }}</td>
            <td>{{ e.filename or "N/A" }}</td>
            <td>{{ e.type or "N/A" }}</td>
            <td>
              {% if e.event == "automated" %}{{ e.action }}{% if e.destination %} &rarr; {{ e.destination }}{% endif %}
              {% elif e.event == "scanned" %}{{ "needs review" if e.needs_review else "clean read" }}
              {% endif %}
            </td>
            <td>
              {% if e.event == "automated" %}<span class="{{ 'audit-ok' if e.ok else 'audit-fail' }}">{{ "OK" if e.ok else "Failed" }}</span>
              {% elif e.event == "scanned" %}<span class="{{ 'audit-fail' if e.error else 'audit-ok' }}">{{ "Error" if e.error else "OK" }}</span>
              {% endif %}
            </td>
            <td>{{ e.ip or "N/A" }}</td>
          </tr>
          {% endfor %}
        </table>
      </div>
      {% else %}
      <p class="audit-empty">Nothing recorded yet.</p>
      {% endif %}
    </div>
  </main>

  <footer><p>Showing the most recent {{ entries|length }} entr{{ "y" if entries|length == 1 else "ies" }}. <a href="/trust-report" style="color:inherit;">View the trust &amp; custody report</a>.</p></footer>
</div>
</body>
</html>
"""

TRUST_STYLE = """
  .trust-claim { background: var(--cyan-tint); border: 1px solid var(--line); border-radius: 14px;
                 padding: 20px 22px; font-size: 0.98rem; line-height: 1.6; margin-bottom: 24px; }
  .trust-section { margin-bottom: 24px; }
  .trust-section h2 { font-size: 1rem; margin: 0 0 10px; }
  .trust-stats { display: flex; gap: 28px; flex-wrap: wrap; }
  .trust-stats .stat-value { font-size: 1.3rem; font-weight: 700; color: var(--navy); display: block; }
  .trust-stats .stat-label { font-size: 0.78rem; color: var(--text-soft); }
  .trust-security-list { margin: 0; padding-left: 20px; color: var(--text-soft); font-size: 0.9rem; line-height: 1.7; }
  .trust-disclaimer { font-size: 0.8rem; color: var(--text-soft); border-top: 1px solid var(--line);
                       padding-top: 16px; margin-top: 24px; line-height: 1.6; }
  .trust-meta { color: var(--text-soft); font-size: 0.85rem; margin-bottom: 28px; }
  @media print {
    header, .trust-print-btn, footer { display: none !important; }
    body { background: #fff; }
    .card { box-shadow: none; border: none; }
  }
"""

TRUST_REPORT_PAGE = """
<!doctype html>
<html lang="en">
<head>""" + HEAD + """<title>Nuru | Trust &amp; custody report</title><style>""" + BASE_STYLE + TRUST_STYLE + """</style></head>
<body>
<div class="page">
""" + HEADER_NAV + """
  <main>
    <h1>Trust &amp; custody report</h1>
    <p class="trust-meta">Generated {{ generated_at }} (UTC) &middot; covers this installation's audit log
      {% if audit_summary.first_entry_at %}from {{ audit_summary.first_entry_at }} to {{ audit_summary.last_entry_at }}{% else %}(no activity recorded yet){% endif %}.</p>

    <div class="trust-claim">
      Every document processed by this Nuru installation was extracted locally, using a
      self-hosted, hand-built model (pure NumPy: no PyTorch, no TensorFlow, no pretrained
      weights). No document content or extracted field value was sent to any third-party AI
      service, cloud OCR API, or external model as part of extraction. This claim covers
      extraction only; if a document was later sent to a webhook or email address, that was
      a destination this installation's user explicitly chose in the Automate step, not
      something implied by this statement.
    </div>

    <div class="card trust-section">
      <h2>Activity summary</h2>
      <div class="trust-stats">
        <div><span class="stat-value">{{ audit_summary.total_scanned }}</span><span class="stat-label">documents scanned</span></div>
        <div><span class="stat-value">{{ audit_summary.total_automated }}</span><span class="stat-label">automations run</span></div>
        <div><span class="stat-value">{{ ledger_totals.count }}</span><span class="stat-label">entries in the ledger</span></div>
        <div><span class="stat-value">{{ "$%.2f"|format(ledger_totals.sum_amount) if ledger_totals.sum_amount is not none else "N/A" }}</span><span class="stat-label">recognized ledger total</span></div>
      </div>
      {% if audit_summary.by_action %}
      <p style="color:var(--text-soft); font-size:0.85rem; margin-top:14px; margin-bottom:0;">
        By action: {% for action, count in audit_summary.by_action.items() %}{{ action }} ({{ count }}){% if not loop.last %}, {% endif %}{% endfor %}
      </p>
      {% endif %}
    </div>

    <div class="card trust-section">
      <h2>Security posture</h2>
      <ul class="trust-security-list">
        <li>Upload size capped per request</li>
        <li>Webhook destinations checked before sending (link-local, multicast, and reserved ranges always blocked; redirects never followed)</li>
        <li>Archive destinations restricted to an administrator-approved allowlist</li>
        <li>CSRF protection on every state-changing request</li>
        <li>Rate limiting on every route</li>
        <li>Documents scanned but never carried through review are auto-purged on a bounded schedule</li>
        <li>The audit log is metadata-only: it records that an action happened and whether it succeeded, never the extracted field values themselves</li>
      </ul>
    </div>

    <button type="button" class="btn btn-ghost trust-print-btn" onclick="window.print()">Print / Save as PDF</button>

    <p class="trust-disclaimer">
      This report is generated from this installation's own local audit trail and reflects this
      installation only. It is not an independent third-party attestation. See
      docs/soc2-readiness.md in the Nuru repository for what a formal compliance certification
      would additionally require.
    </p>
  </main>
</div>
</body>
</html>
"""

LEDGER_STYLE = """
  .ledger-summary { display: flex; gap: 28px; flex-wrap: wrap; margin-bottom: 20px; }
  .ledger-stat .stat-value { font-size: 1.4rem; font-weight: 700; color: var(--navy); display: block; }
  .ledger-stat .stat-label { font-size: 0.78rem; color: var(--text-soft); }
  .ledger-filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .ledger-filters input[type=text], .ledger-filters input[type=date], .ledger-filters select {
    padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; font-family: inherit;
    font-size: 0.85rem; color: var(--text); background: #fff;
  }
  .ledger-filters input[type=text] { flex: 1; min-width: 160px; }
  .ledger-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  .ledger-table th { text-align: left; padding: 8px 10px; color: var(--text-soft); font-weight: 600;
                      border-bottom: 1px solid var(--line); white-space: nowrap; }
  .ledger-table td { padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
  .ledger-table tr:last-child td { border-bottom: none; }
  .ledger-amount { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .ledger-empty { color: var(--text-soft); padding: 24px 0; text-align: center; }
  .ledger-fields-detail summary { cursor: pointer; color: var(--text-soft); font-size: 0.8rem; }
  .ledger-fields-detail dl { margin: 8px 0 0; font-size: 0.8rem; }
  .ledger-fields-detail dt { color: var(--text-soft); }
  .ledger-fields-detail dd { margin: 0 0 6px; }
  .ledger-delete-btn { background: none; border: none; color: var(--text-soft); cursor: pointer;
                        font-size: 0.8rem; text-decoration: underline; padding: 0; }
  .ledger-delete-btn:hover { color: var(--error); }
  .ledger-more { display: block; text-align: center; margin-top: 16px; color: var(--navy);
                 font-weight: 600; text-decoration: none; font-size: 0.9rem; }
"""

LEDGER_PAGE = """
<!doctype html>
<html lang="en">
<head>""" + HEAD + """<title>Nuru | Ledger</title><style>""" + BASE_STYLE + LEDGER_STYLE + """</style></head>
<body>
<div class="page">
""" + HEADER_NAV + """
  <main>
    <h1>Ledger</h1>
    <p class="tagline">Documents you've saved stay here, in this installation's own local database, searchable and totaled. Nothing here is sent anywhere. <a href="/trust-report" style="color:inherit;">View the trust &amp; custody report</a>.</p>

    <div class="ledger-summary">
      <div class="ledger-stat"><span class="stat-value">{{ totals.count }}</span><span class="stat-label">entries</span></div>
      <div class="ledger-stat"><span class="stat-value">{{ "$%.2f"|format(totals.sum_amount) if totals.sum_amount is not none else "N/A" }}</span><span class="stat-label">total (recognized amounts)</span></div>
      {% if totals.unrecognized_amount_count %}
      <div class="ledger-stat"><span class="stat-value">{{ totals.unrecognized_amount_count }}</span><span class="stat-label">without a recognized amount</span></div>
      {% endif %}
    </div>

    <form class="ledger-filters" method="get" action="/ledger">
      <input type="text" name="q" placeholder="Search filename, counterparty, category" value="{{ q or '' }}">
      <select name="type">
        <option value="">All types</option>
        {% for t in ["Invoice", "Receipt", "Statement"] %}
        <option value="{{ t }}" {{ "selected" if doc_type == t else "" }}>{{ t }}</option>
        {% endfor %}
      </select>
      <select name="category">
        <option value="">All categories</option>
        {% for c in categories %}
        <option value="{{ c }}" {{ "selected" if category == c else "" }}>{{ c }}</option>
        {% endfor %}
      </select>
      <input type="date" name="from" value="{{ date_from or '' }}">
      <input type="date" name="to" value="{{ date_to or '' }}">
      <button type="submit" class="btn btn-ghost btn-small">Filter</button>
    </form>

    <div class="card">
      {% if entries %}
      <div class="table-scroll">
        <table class="ledger-table">
          <tr><th>Date</th><th>Document</th><th>Type</th><th>Counterparty</th><th>Category</th><th>Amount</th><th>Fields</th><th></th></tr>
          {% for e in entries %}
          <tr>
            <td>{{ e.document_date or "N/A" }}</td>
            <td>{{ e.original_filename }}</td>
            <td>{{ e.doc_type or "N/A" }}</td>
            <td>{{ e.counterparty or "N/A" }}</td>
            <td>{{ e.category or "N/A" }}</td>
            <td class="ledger-amount">{{ e.amount_raw or "N/A" }}</td>
            <td>
              <details class="ledger-fields-detail">
                <summary>{{ e.fields_json|from_json|length }} field(s)</summary>
                <dl>
                  {% for f in e.fields_json|from_json %}
                  <dt>{{ f.label }}</dt><dd>{{ f.value or "Not provided" }}</dd>
                  {% endfor %}
                </dl>
              </details>
            </td>
            <td>
              <form method="post" action="/ledger/{{ e.id }}/delete" onsubmit="return confirm('Delete this ledger entry? This can&#39;t be undone.');">
                <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
                <button type="submit" class="ledger-delete-btn">Delete</button>
              </form>
            </td>
          </tr>
          {% endfor %}
        </table>
      </div>
      {% if has_more %}
      <a class="ledger-more" href="/ledger?{{ query_string }}&offset={{ offset + entries|length }}">Older entries &rsaquo;</a>
      {% endif %}
      {% else %}
      <p class="ledger-empty">Nothing saved to the ledger yet. Approve a document and choose "Save to ledger" to start one.</p>
      {% endif %}
    </div>
  </main>
</div>
</body>
</html>
"""

REVIEW_PAGE = """
<!doctype html>
<html lang="en">
<head>
""" + HEAD + """
<title>Nuru | Review</title>
<style>""" + BASE_STYLE + REVIEW_STYLE + """</style>
</head>
<body>
<div class="page">
""" + HEADER_NAV + """
  <div class="stepper">
    <span class="step active" id="step-pill-review">Review</span>
    <span class="step-arrow">&rsaquo;</span>
    <span class="step" id="step-pill-automate">Automate</span>
    <span class="step-arrow">&rsaquo;</span>
    <span class="step" id="step-pill-done">Done</span>
  </div>

  <main>
    {% if record.error %}
      <div class="card">
        <div class="doc-header"><span class="doc-name">{{ record.original_filename }}</span></div>
        <p style="color:var(--text-soft); margin:0;">{{ record.notes }}</p>
        <div class="actions-row">
          {% if next_token %}
          <a class="btn btn-primary" style="text-decoration:none; display:inline-flex;" href="/review/{{ next_token }}">Skip to next document</a>
          {% else %}
          <a class="btn btn-primary" style="text-decoration:none; display:inline-flex;" href="/">Back to upload</a>
          {% endif %}
        </div>
      </div>
    {% else %}
    <div class="card" id="step-review">
      <div class="doc-header">
        <span class="doc-name">{{ record.original_filename }}</span>
        <span class="badge badge-type">{{ record.type }}</span>
      </div>

      <div id="field-grid">
        {% for f in record.fields %}
        <div class="field-row {{ 'flagged' if (not f.value) or (f.confidence is not none and f.confidence < threshold) else '' }}" data-row>
          <div class="field-label-group">
            {% set presets = preset_labels.get(f.key, []) %}
            {% if presets %}
            <select class="label-preset">
              <option value="">Alternate terms…</option>
              {% for p in presets %}<option value="{{ p }}">{{ p }}</option>{% endfor %}
            </select>
            {% endif %}
            <input type="text" class="label-input" value="{{ f.label }}">
          </div>
          <input type="text" class="value-input" value="{{ f.value }}" placeholder="Not found">
          <button type="button" class="remove-btn" title="Remove field" onclick="this.closest('[data-row]').remove()">&times;</button>
        </div>
        {% endfor %}
      </div>

      <button type="button" class="add-field-btn" id="add-field-btn">+ Add custom field</button>

      {% if record.notes %}<p style="color:var(--text-soft); font-size:0.85rem; margin-top:16px;">{{ record.notes }}</p>{% endif %}

      <div class="actions-row">
        <button type="button" class="btn btn-primary" id="continue-btn">Continue to automation</button>
      </div>
    </div>

    <div class="card" id="step-automate" style="display:none;">
      <h1 style="font-size:1.2rem; margin-top:0;">Send this document on</h1>
      <div class="action-grid">
        <div class="action-card" data-action="ledger">
          <span class="a-title">Save to ledger</span>
          <span class="a-sub">Add it to Nuru's built-in bookkeeping record</span>
        </div>
        <div class="action-card" data-action="webhook">
          <span class="a-title">Send to a webhook</span>
          <span class="a-sub">Zapier, Make, or any endpoint you run</span>
        </div>
        <div class="action-card" data-action="email">
          <span class="a-title">Email it</span>
          <span class="a-sub">A formatted summary to an address you choose</span>
        </div>
        <div class="action-card" data-action="archive">
          <span class="a-title">Archive the file</span>
          <span class="a-sub">Rename and store the original PDF</span>
        </div>
        <div class="action-card" data-action="download">
          <span class="a-title">Download as Excel</span>
          <span class="a-sub">Just get the spreadsheet</span>
        </div>
      </div>

      <div class="action-param" id="param-ledger" style="display:none;">
        <select id="param-ledger-preset">
          <option value="">Choose a category…</option>
          {% for c in ledger_categories %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
        </select>
        <input type="text" id="param-ledger-input" placeholder="Category (optional)" value="" style="margin-top:8px;">
      </div>
      <div class="action-param" id="param-webhook" style="display:none;">
        <input type="text" id="param-webhook-input" placeholder="https://hooks.example.com/..." value="{{ default_webhook_url }}">
      </div>
      <div class="action-param" id="param-email" style="display:none;">
        <input type="text" id="param-email-input" placeholder="accounts-payable@company.com" value="{{ default_email_to }}">
      </div>
      <div class="action-param" id="param-archive" style="display:none;">
        <select id="param-archive-input">
          {% for dir in allowed_archive_dirs %}
          <option value="{{ dir }}">{{ dir }}</option>
          {% endfor %}
        </select>
      </div>

      <div class="actions-row">
        <button type="button" class="btn btn-ghost" id="preview-btn" disabled>Preview what will be sent</button>
      </div>
      <pre id="preview-box" class="preview-box" style="display:none;"></pre>

      <div class="actions-row">
        <button type="button" class="btn btn-ghost" id="back-btn">Back</button>
        <button type="button" class="btn btn-primary" id="run-btn" disabled>Choose an option above</button>
      </div>
      <div class="actions-row">
        <button type="button" class="btn btn-ghost" id="discard-btn">Discard, don't send anywhere</button>
      </div>
      <div id="automate-error"></div>
    </div>

    <div class="card" id="step-result" style="display:none;">
      <p id="result-message" style="font-size:1.05rem; font-weight:600;"></p>
      <div id="download-filename-group" style="display:none;">
        <label for="download-filename-input" style="display:block; font-size:0.82rem; color:var(--text-soft); margin-bottom:6px;">File name</label>
        <input type="text" id="download-filename-input" style="width:100%; margin-bottom:12px; padding:10px 12px; border:1px solid var(--line); border-radius:10px; font-family:inherit; font-size:0.92rem; color:var(--text); background:#fff;">
      </div>
      <a class="btn btn-primary" id="download-link" style="display:none; text-decoration:none;" href="#">Download the spreadsheet</a>
      <a class="next-link" id="next-link" href="#" style="display:none;">Next document &rsaquo;</a>
      <a class="next-link" id="upload-more-link" href="/" style="display:none;">Upload more documents</a>
    </div>
    {% endif %}
  </main>

  <footer><p>Once you send this document on, the source file is permanently deleted from Nuru's cache.</p></footer>
</div>

<script>
  const token = {{ token|tojson }};
  const nextToken = {{ next_token|tojson }};
  const csrfToken = {{ csrf_token()|tojson }};

  const stepReview = document.getElementById('step-review');
  const stepAutomate = document.getElementById('step-automate');
  const stepResult = document.getElementById('step-result');
  const pillReview = document.getElementById('step-pill-review');
  const pillAutomate = document.getElementById('step-pill-automate');
  const pillDone = document.getElementById('step-pill-done');

  document.getElementById('add-field-btn')?.addEventListener('click', () => {
    const grid = document.getElementById('field-grid');
    const row = document.createElement('div');
    row.className = 'field-row';
    row.setAttribute('data-row', '');
    row.innerHTML = `
      <div class="field-label-group">
        <input type="text" class="label-input" placeholder="Field name" value="">
      </div>
      <input type="text" class="value-input" placeholder="Value" value="">
      <button type="button" class="remove-btn" title="Remove field" onclick="this.closest('[data-row]').remove()">&times;</button>
    `;
    grid.appendChild(row);
    row.querySelector('.label-input').focus();
  });

  document.querySelectorAll('.label-preset').forEach(select => {
    select.addEventListener('change', () => {
      if (!select.value) return;
      const labelInput = select.closest('.field-label-group').querySelector('.label-input');
      labelInput.value = select.value;
    });
  });

  document.getElementById('continue-btn')?.addEventListener('click', () => {
    stepReview.style.display = 'none';
    stepAutomate.style.display = 'block';
    pillReview.classList.remove('active');
    pillAutomate.classList.add('active');
  });
  document.getElementById('back-btn')?.addEventListener('click', () => {
    stepAutomate.style.display = 'none';
    stepReview.style.display = 'block';
    pillAutomate.classList.remove('active');
    pillReview.classList.add('active');
  });

  let selectedAction = null;
  const paramBoxes = { ledger: document.getElementById('param-ledger'),
                        webhook: document.getElementById('param-webhook'),
                        email: document.getElementById('param-email'),
                        archive: document.getElementById('param-archive'),
                        download: null };
  const runBtn = document.getElementById('run-btn');
  const previewBtn = document.getElementById('preview-btn');
  const previewBox = document.getElementById('preview-box');

  document.getElementById('param-ledger-preset')?.addEventListener('change', function () {
    if (!this.value) return;
    document.getElementById('param-ledger-input').value = this.value;
  });

  const RUN_LABELS = { download: 'Get the spreadsheet', ledger: 'Save to ledger' };

  document.querySelectorAll('.action-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.action-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      selectedAction = card.getAttribute('data-action');
      Object.values(paramBoxes).forEach(box => { if (box) box.style.display = 'none'; });
      if (paramBoxes[selectedAction]) paramBoxes[selectedAction].style.display = 'block';
      runBtn.disabled = false;
      runBtn.textContent = RUN_LABELS[selectedAction] || 'Run automation';
      previewBtn.disabled = false;
      previewBox.style.display = 'none';
    });
  });

  function collectFields() {
    const rows = document.querySelectorAll('#field-grid [data-row]');
    const fields = [];
    rows.forEach(row => {
      const label = row.querySelector('.label-input').value.trim();
      const value = row.querySelector('.value-input').value.trim();
      if (label) fields.push({ label, value });
    });
    return fields;
  }

  function paramFor(action) {
    if (action === 'ledger') return document.getElementById('param-ledger-input').value.trim();
    if (action === 'webhook') return document.getElementById('param-webhook-input').value.trim();
    if (action === 'email') return document.getElementById('param-email-input').value.trim();
    if (action === 'archive') return document.getElementById('param-archive-input').value.trim();
    return '';
  }

  function runAutomation(action) {
    const errBox = document.getElementById('automate-error');
    errBox.innerHTML = '';
    runBtn.disabled = true;
    fetch('/automate/' + token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
      body: JSON.stringify({ fields: collectFields(), action: action, param: paramFor(action) }),
    })
      .then(r => r.json())
      .then(data => {
        if (!data.ok) {
          errBox.innerHTML = '<div class="notice notice-error">' + data.message + '</div>';
          runBtn.disabled = false;
          return;
        }
        stepAutomate.style.display = 'none';
        stepResult.style.display = 'block';
        pillAutomate.classList.remove('active');
        pillDone.classList.add('active');
        document.getElementById('result-message').textContent = data.message;
        if (data.download_url) {
          const dl = document.getElementById('download-link');
          const nameGroup = document.getElementById('download-filename-group');
          const nameInput = document.getElementById('download-filename-input');
          nameInput.value = data.download_filename || 'Nuru - Export';
          const refreshHref = () => {
            dl.href = data.download_url + '?name=' + encodeURIComponent(nameInput.value.trim());
          };
          nameInput.oninput = refreshHref;
          refreshHref();
          nameGroup.style.display = 'block';
          dl.style.display = 'inline-flex';
        }
        if (data.next_token) {
          const nl = document.getElementById('next-link');
          nl.href = '/review/' + data.next_token;
          nl.style.display = 'block';
        } else {
          document.getElementById('upload-more-link').style.display = 'block';
        }
      })
      .catch(() => {
        errBox.innerHTML = '<div class="notice notice-error">Something went wrong. Please try again.</div>';
        runBtn.disabled = false;
      });
  }

  function previewAutomation() {
    if (!selectedAction) return;
    const errBox = document.getElementById('automate-error');
    errBox.innerHTML = '';
    previewBtn.disabled = true;
    previewBtn.textContent = 'Loading…';
    fetch('/preview/' + token, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
      body: JSON.stringify({ fields: collectFields(), action: selectedAction }),
    })
      .then(r => r.json())
      .then(data => {
        previewBtn.disabled = false;
        previewBtn.textContent = 'Preview what will be sent';
        if (!data.ok) {
          errBox.innerHTML = '<div class="notice notice-error">' + data.message + '</div>';
          return;
        }
        previewBox.textContent = data.preview;
        previewBox.style.display = 'block';
      })
      .catch(() => {
        previewBtn.disabled = false;
        previewBtn.textContent = 'Preview what will be sent';
        errBox.innerHTML = '<div class="notice notice-error">Something went wrong. Please try again.</div>';
      });
  }

  runBtn?.addEventListener('click', () => { if (selectedAction) runAutomation(selectedAction); });
  previewBtn?.addEventListener('click', previewAutomation);
  document.getElementById('discard-btn')?.addEventListener('click', () => runAutomation('discard'));
</script>
</body>
</html>
"""


@app.before_request
def _cleanup_stale_reviews():
    _purge_stale_pending()


@app.before_request
def _require_access_password():
    """Gate every route behind HTTP Basic Auth, but only if NURU_ACCESS_PASSWORD
    is actually set. Unset (the default for local-only use) means no gate at
    all; this only matters once Nuru is reachable from beyond localhost.

    /healthz is always exempt: a load balancer or uptime monitor hitting it
    has no way to carry a password, and health checks failing because
    nobody configured credentials for the monitoring tool defeats the
    point of having one."""
    if request.path == "/healthz":
        return None
    required = os.environ.get("NURU_ACCESS_PASSWORD")
    if not required:
        return None
    auth = request.authorization
    if auth and auth.password == required:
        return None
    return Response(
        "Authentication required.", 401,
        {"WWW-Authenticate": 'Basic realm="Nuru"'},
    )


# The templates use inline <style>/<script> blocks rather than nonces, so
# style-src/script-src still need 'unsafe-inline', which is a real gap in
# what this CSP protects against, not a solved problem. Moving to
# per-request nonces on every inline block would close it; everything else
# here (no third-party script hosts, no embedding, no unrelated connect
# targets) is enforced for real.
_CSP = (
    "default-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "script-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)


@app.after_request
def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"] = _CSP
    # Only takes effect once actually served over HTTPS (e.g. behind a
    # reverse proxy); browsers ignore it over plain HTTP.
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.errorhandler(Exception)
def _handle_unexpected_error(exc):
    """A final backstop: anything that reaches here escaped every
    try/except in the actual route logic. Log it persistently (and to
    Sentry if configured) instead of it only ever existing in Flask's
    default traceback page, then show a plain, non-leaking message.

    HTTPException (404, 401 from the auth gate, 429 from rate limiting,
    400 from CSRF) is deliberately passed straight through to Flask's own
    handling instead: those are expected, already-correct responses, not
    failures worth logging as errors."""
    if isinstance(exc, HTTPException):
        return exc.get_response()
    errors.report_exception(exc, path=request.path, method=request.method)
    if request.path.startswith("/automate/"):
        return jsonify(ok=False, message="Something went wrong on our end. Please try again."), 500
    return "Something went wrong on our end. Please try again.", 500


@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_PAGE, error=None, active="upload")


@app.route("/scan", methods=["POST"])
@limiter.limit("10 per minute")
def scan():
    uploaded_files = [f for f in request.files.getlist("pdfs") if f and f.filename]
    if not uploaded_files:
        return render_template_string(UPLOAD_PAGE, error="Please choose at least one PDF file.", active="upload")

    try:
        model, vocab = get_model()
    except Exception:
        return render_template_string(UPLOAD_PAGE, error="Something went wrong on our end. Please try again in a moment.", active="upload")

    batch_tokens = []
    for uploaded in uploaded_files:
        filename = secure_filename(uploaded.filename)
        token = uuid.uuid4().hex
        pdf_path = os.path.join(_RESULTS_DIR, f"{token}_{filename}")
        uploaded.save(pdf_path)

        result = process_invoice(model, vocab, pdf_path, DEFAULT_CONFIDENCE_THRESHOLD)
        _PENDING[token] = {
            "pdf_path": None if result["Error"] else pdf_path,
            "original_filename": filename,
            "type": result["Type"],
            "fields": result["Fields"],
            "notes": result["Notes"],
            "needs_review": result["NeedsReview"],
            "error": result["Error"],
            "done": False,
            "batch_tokens": None,
            "batch_index": None,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if result["Error"]:
            _purge_cached_pdf(pdf_path)  # nothing to review; no reason to keep it cached
        audit.record(
            "scanned", ip=request.remote_addr, filename=filename,
            type=result["Type"], error=result["Error"], needs_review=result["NeedsReview"],
        )
        batch_tokens.append(token)

    for i, t in enumerate(batch_tokens):
        _PENDING[t]["batch_tokens"] = batch_tokens
        _PENDING[t]["batch_index"] = i
    _save_state()

    return redirect(f"/review/{batch_tokens[0]}")


@app.route("/review/<token>", methods=["GET"])
def review(token):
    record = _PENDING.get(token)
    if record is None:
        return render_template_string(EXPIRED_PAGE, active="upload"), 404
    next_token = _next_pending_token(record)
    if record["error"] and not record["done"]:
        # A failed scan has nothing to review or automate; viewing this page
        # once is the whole "completion" of it, so it won't be revisited.
        record["done"] = True
        _save_state()
    return render_template_string(
        REVIEW_PAGE, token=token, record=record, preset_labels=PRESET_LABELS,
        threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        allowed_archive_dirs=automation.allowed_archive_dirs(),
        default_webhook_url=automation.default_webhook_url(),
        default_email_to=automation.default_email_to(),
        ledger_categories=LEDGER_CATEGORIES,
        next_token=next_token,
    )


def _parse_rows(data):
    return [
        {"label": (r.get("label") or "").strip(), "value": (r.get("value") or "").strip()}
        for r in data.get("fields", [])
        if (r.get("label") or "").strip()
    ]


@app.route("/preview/<token>", methods=["POST"])
@limiter.limit("30 per minute")
def preview(token):
    """Shows exactly what a chosen action would send or produce, without
    actually sending or producing it: no webhook POST, no email, no file
    move. Lets someone setting up a Zap/Scenario see real field names
    before they build anything around them, addressed directly at "no
    self-serve integration story": pasting a webhook URL only gets you so
    far if you can't see what's coming out the other end."""
    record = _PENDING.get(token)
    if record is None:
        return jsonify(ok=False, message="This review session has expired."), 404

    data = request.get_json(silent=True) or {}
    rows = _parse_rows(data)
    action = data.get("action")

    if action == "ledger":
        counterparty = next((r["value"] for r in rows if r["label"] in ledger._COUNTERPARTY_LABELS and r["value"]), None)
        amount = next((r["value"] for r in rows if r["label"] in ledger._AMOUNT_LABELS and r["value"]), None)
        summary = f"{counterparty or '(no counterparty recognized)'}: {amount or '(no amount recognized)'}"
        return jsonify(ok=True, preview=f"Would be saved to the ledger as:\n{summary}\n\nAll {len(rows)} approved field(s) are kept regardless of whether they were recognized above.")
    if action == "webhook":
        payload = {row["label"]: row["value"] for row in rows}
        payload["_document"] = record["original_filename"]
        payload["_type"] = record["type"]
        return jsonify(ok=True, preview=json.dumps(payload, indent=2))
    if action == "email":
        subject = f"{record['type'] or 'Document'} details: {record['original_filename']}"
        body = "\n".join(f"{r['label']}: {r['value'] or 'Not provided'}" for r in rows)
        return jsonify(ok=True, preview=f"Subject: {subject}\n\n{body}")
    if action == "archive":
        filename = automation.build_archive_filename(rows, record["original_filename"])
        return jsonify(ok=True, preview=f"Would be saved as:\n{filename}")
    if action == "download":
        return jsonify(ok=True, preview="Generates an Excel file with the fields below, one row.")
    return jsonify(ok=False, message="Choose an option above first."), 400


@app.route("/automate/<token>", methods=["POST"])
@limiter.limit("20 per minute")
def automate(token):
    record = _PENDING.get(token)
    if record is None:
        return jsonify(ok=False, message="This review session has expired."), 404

    data = request.get_json(silent=True) or {}
    rows = _parse_rows(data)
    action = data.get("action")
    param = (data.get("param") or "").strip()

    download_url = None
    download_filename = None
    if action == "ledger":
        ok, message = ledger.save_entry(rows, record, token, category=param)
    elif action == "webhook":
        payload = {row["label"]: row["value"] for row in rows}
        payload["_document"] = record["original_filename"]
        payload["_type"] = record["type"]
        ok, message = automation.send_webhook(param, payload)
    elif action == "email":
        subject = f"{record['type'] or 'Document'} details: {record['original_filename']}"
        ok, message = automation.send_email(param, subject, rows)
    elif action == "archive":
        if not record["pdf_path"]:
            ok, message = False, "The source file isn't available to archive anymore."
        else:
            filename = automation.build_archive_filename(rows, record["original_filename"])
            ok, message = automation.archive_file(record["pdf_path"], param, filename)
            if ok:
                record["pdf_path"] = None  # archive_file already moved it
    elif action == "download":
        xlsx_path = os.path.join(_RESULTS_DIR, f"{token}.xlsx")
        excel_row = {
            "File": record["original_filename"], "Type": record["type"],
            "Fields": [{"label": r["label"], "value": r["value"], "confidence": None} for r in rows],
            "NeedsReview": False, "Notes": "Reviewed and approved in Nuru.", "Error": False,
        }
        write_excel([excel_row], xlsx_path)
        ok, message = True, "Ready to download."
        download_url = f"/download/{token}"
        download_filename = _default_download_filename(record, token)
    elif action == "discard":
        ok, message = True, "Discarded. Nothing was sent anywhere."
    else:
        return jsonify(ok=False, message="That's not a recognized action."), 400

    if ok:
        _purge_cached_pdf(record.get("pdf_path"))
        record["done"] = True
        _save_state()

    destination = param if action in ("webhook", "email", "archive") else None
    audit.record(
        "automated", ip=request.remote_addr, filename=record["original_filename"],
        type=record["type"], action=action, destination=destination, ok=ok, message=message,
    )

    return jsonify(
        ok=ok, message=message, download_url=download_url, download_filename=download_filename,
        next_token=_next_pending_token(record) if ok else None,
    )


@app.route("/download/<token>", methods=["GET"])
def download(token):
    if not token.replace("-", "").isalnum():
        return "Invalid link", 400
    xlsx_path = os.path.join(_RESULTS_DIR, f"{token}.xlsx")
    if not os.path.exists(xlsx_path):
        return "This link has expired.", 404
    record = _PENDING.get(token)
    # A name built only from the source filename repeats every time the same
    # document is scanned and downloaded again (a very normal thing to do
    # while testing, or re-running a document). When that happens, the
    # browser silently appends its own "(1)"/"(2)" to avoid overwriting the
    # earlier download, and if that earlier file is later moved or deleted,
    # clicking an old download-history entry for it fails with a
    # "Windows cannot find" style error that has nothing to do with this
    # request. Folding in a slice of the (already unique per scan) token
    # into the default keeps every download's filename distinct on its own;
    # the ?name= query param lets someone override it outright before the
    # file is actually requested, from the filename field on the result screen.
    default_name = _default_download_filename(record, token)
    requested_name = request.args.get("name")
    filename_base = _sanitize_download_filename(requested_name, default_name) if requested_name else default_name
    return send_file(xlsx_path, as_attachment=True, download_name=f"{filename_base}.xlsx")


@app.route("/audit", methods=["GET"])
def audit_trail():
    return render_template_string(AUDIT_PAGE, entries=audit.read_recent(), active="audit")


@app.route("/ledger", methods=["GET"])
def ledger_view():
    q = request.args.get("q") or None
    doc_type = request.args.get("type") or None
    category = request.args.get("category") or None
    date_from = request.args.get("from") or None
    date_to = request.args.get("to") or None
    offset = max(0, int(request.args.get("offset") or 0))

    entries, has_more = ledger.list_entries(
        q=q, doc_type=doc_type, category=category, date_from=date_from, date_to=date_to, offset=offset,
    )
    totals = ledger.get_totals(q=q, doc_type=doc_type, category=category, date_from=date_from, date_to=date_to)
    query_string = urlencode({k: v for k, v in
                               [("q", q), ("type", doc_type), ("category", category), ("from", date_from), ("to", date_to)]
                               if v})

    return render_template_string(
        LEDGER_PAGE, entries=entries, totals=totals, has_more=has_more, offset=offset,
        categories=ledger.distinct_categories(), q=q, doc_type=doc_type, category=category,
        date_from=date_from, date_to=date_to, query_string=query_string, active="ledger",
    )


@app.route("/ledger/<int:entry_id>/delete", methods=["POST"])
@limiter.limit("30 per minute")
def ledger_delete(entry_id):
    ledger.delete_entry(entry_id)
    return redirect(request.referrer or "/ledger")


@app.route("/trust-report", methods=["GET"])
def trust_report():
    return render_template_string(
        TRUST_REPORT_PAGE,
        audit_summary=audit.summarize(),
        ledger_totals=ledger.get_totals(),
        generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        active=None,
    )


@app.route("/healthz", methods=["GET"])
@limiter.exempt
def healthz():
    """For a load balancer or uptime monitor, not a person. Checks the one
    thing that actually determines whether Nuru can do its job: is the
    trained model loaded. Deliberately doesn't touch the filesystem cache
    or any automation backend; those failing shouldn't flip the whole
    service to "down" for a check that's meant to be cheap and frequent."""
    if _model is None or _vocab is None:
        return jsonify(status="unhealthy", model_loaded=False), 503
    return jsonify(status="ok", model_loaded=True), 200


get_model()  # fail fast if the model isn't trained yet, whether run directly or via a WSGI server

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
