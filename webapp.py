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

import json
import os
import uuid

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_file
from werkzeug.utils import secure_filename

import automation
from app import (
    DEFAULT_CONFIDENCE_THRESHOLD, VOCAB_PATH, WEIGHTS_PATH,
    process_invoice, write_excel,
)
from engine.model import TokenClassifier
from engine.tokenizer import Vocabulary

app = Flask(__name__)

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
        app.logger.warning(f"could not save pending state: {exc}")


def _load_state():
    global _PENDING
    if not os.path.exists(_STATE_PATH):
        return
    try:
        with open(_STATE_PATH, "r", encoding="utf-8") as f:
            _PENDING = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        app.logger.warning(f"could not load pending state, starting fresh: {exc}")
        _PENDING = {}


_load_state()

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
        app.logger.warning(f"could not purge cached PDF {pdf_path}: {exc}")


def _next_pending_token(record):
    tokens = record["batch_tokens"]
    for t in tokens[record["batch_index"] + 1:]:
        other = _PENDING.get(t)
        if other and not other["done"]:
            return t
    return None


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
  header { display: flex; align-items: center; gap: 10px; margin-bottom: 36px; }
  .logo-mark { flex-shrink: 0; }
  .logo-word { font-weight: 800; font-size: 1.2rem; letter-spacing: -0.01em; color: var(--navy); }
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
  .action-param input[type=text] { width: 100%; padding: 10px 12px; border: 1px solid var(--line);
                                     border-radius: 10px; font-family: inherit; font-size: 0.92rem; }

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
  <header>""" + LOGO_SVG + """<span class="logo-word">Nuru</span></header>
  <main>
    <h1>Give Nuru your documents.</h1>
    <p class="tagline">Drop in invoices, receipts, or statements. Nuru reads each one, then lets you review and approve every field before anything leaves the app.</p>

    <form method="post" action="/scan" enctype="multipart/form-data" id="upload-form">
      <label class="dropzone" id="dropzone" for="pdf-input">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none">
          <path d="M12 3v12M12 3l-4 4M12 3l4 4" stroke="#16205c" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M4 16v2.5A2.5 2.5 0 0 0 6.5 21h11a2.5 2.5 0 0 0 2.5-2.5V16" stroke="#16205c" stroke-width="1.8" stroke-linecap="round"/>
        </svg>
        <span class="dz-title" id="dz-label">Drop your documents here</span>
        <span class="dz-sub" id="dz-sub">or click to browse. One or more PDF files</span>
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
  <header>""" + LOGO_SVG + """<span class="logo-word">Nuru</span></header>
  <main>
    <h1>This review session is gone.</h1>
    <p class="tagline">It may have already been completed, or it's simply expired. Start again with a fresh upload.</p>
    <a class="btn btn-primary" style="text-decoration:none; display:inline-flex;" href="/">Upload documents</a>
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
  <header>""" + LOGO_SVG + """<span class="logo-word">Nuru</span></header>

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

      <div class="action-param" id="param-webhook" style="display:none;">
        <input type="text" id="param-webhook-input" placeholder="https://hooks.example.com/..." value="{{ default_webhook_url }}">
      </div>
      <div class="action-param" id="param-email" style="display:none;">
        <input type="text" id="param-email-input" placeholder="accounts-payable@company.com" value="{{ default_email_to }}">
      </div>
      <div class="action-param" id="param-archive" style="display:none;">
        <input type="text" id="param-archive-input" placeholder="{{ default_archive_dir }}" value="{{ default_archive_dir }}">
      </div>

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
  const paramBoxes = { webhook: document.getElementById('param-webhook'),
                        email: document.getElementById('param-email'),
                        archive: document.getElementById('param-archive'),
                        download: null };
  const runBtn = document.getElementById('run-btn');

  document.querySelectorAll('.action-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.action-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      selectedAction = card.getAttribute('data-action');
      Object.values(paramBoxes).forEach(box => { if (box) box.style.display = 'none'; });
      if (paramBoxes[selectedAction]) paramBoxes[selectedAction].style.display = 'block';
      runBtn.disabled = false;
      runBtn.textContent = selectedAction === 'download' ? 'Get the spreadsheet' : 'Run automation';
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
      headers: { 'Content-Type': 'application/json' },
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
          dl.href = data.download_url;
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

  runBtn?.addEventListener('click', () => { if (selectedAction) runAutomation(selectedAction); });
  document.getElementById('discard-btn')?.addEventListener('click', () => runAutomation('discard'));
</script>
</body>
</html>
"""


@app.before_request
def _require_access_password():
    """Gate every route behind HTTP Basic Auth, but only if NURU_ACCESS_PASSWORD
    is actually set. Unset (the default for local-only use) means no gate at
    all — this only matters once Nuru is reachable from beyond localhost."""
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


@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_PAGE, error=None)


@app.route("/scan", methods=["POST"])
def scan():
    uploaded_files = [f for f in request.files.getlist("pdfs") if f and f.filename]
    if not uploaded_files:
        return render_template_string(UPLOAD_PAGE, error="Please choose at least one PDF file.")

    try:
        model, vocab = get_model()
    except Exception:
        return render_template_string(UPLOAD_PAGE, error="Something went wrong on our end. Please try again in a moment.")

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
        }
        if result["Error"]:
            _purge_cached_pdf(pdf_path)  # nothing to review; no reason to keep it cached
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
        return render_template_string(EXPIRED_PAGE), 404
    next_token = _next_pending_token(record)
    if record["error"] and not record["done"]:
        # A failed scan has nothing to review or automate; viewing this page
        # once is the whole "completion" of it, so it won't be revisited.
        record["done"] = True
        _save_state()
    return render_template_string(
        REVIEW_PAGE, token=token, record=record, preset_labels=PRESET_LABELS,
        threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        default_archive_dir=automation.DEFAULT_ARCHIVE_DIR,
        default_webhook_url=automation.default_webhook_url(),
        default_email_to=automation.default_email_to(),
        next_token=next_token,
    )


@app.route("/automate/<token>", methods=["POST"])
def automate(token):
    record = _PENDING.get(token)
    if record is None:
        return jsonify(ok=False, message="This review session has expired."), 404

    data = request.get_json(silent=True) or {}
    rows = [
        {"label": (r.get("label") or "").strip(), "value": (r.get("value") or "").strip()}
        for r in data.get("fields", [])
        if (r.get("label") or "").strip()
    ]
    action = data.get("action")
    param = (data.get("param") or "").strip()

    download_url = None
    if action == "webhook":
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
    elif action == "discard":
        ok, message = True, "Discarded. Nothing was sent anywhere."
    else:
        return jsonify(ok=False, message="That's not a recognized action."), 400

    if ok:
        _purge_cached_pdf(record.get("pdf_path"))
        record["done"] = True
        _save_state()

    return jsonify(
        ok=ok, message=message, download_url=download_url,
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
    stem = os.path.splitext(record["original_filename"])[0] if record else "Invoice"
    return send_file(xlsx_path, as_attachment=True, download_name=f"Nuru - {stem}.xlsx")


get_model()  # fail fast if the model isn't trained yet, whether run directly or via a WSGI server

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
