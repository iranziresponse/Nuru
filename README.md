# Nuru

Nuru reads invoices, receipts, and financial statements from PDFs and
turns them into organized records, with a human reviewing and approving
every field before anything leaves the app.

It's built from scratch: the classifier underneath is hand-written NumPy
(embedding lookup, a context window, a ReLU hidden layer, a softmax
output, manually-derived backpropagation) — no PyTorch, no TensorFlow, no
pretrained models.

## How it works

**Scan → Review → Automate**, each its own screen:

1. **Scan** — upload one or more PDFs. Nuru extracts text (falling back to
   OCR if a page has no text layer) and classifies each token.
2. **Review** — Nuru infers what kind of document it is (Invoice, Receipt,
   or Statement) from which fields it actually found, and shows only the
   fields relevant to that kind, labeled in plain business language. Every
   field is editable: rename it, remove it, or add a custom one.
3. **Automate** — send the approved data to a webhook (Zapier, Make, or
   your own endpoint), email it, archive the source PDF under a clean
   name, or just download it as Excel. The source file is permanently
   deleted from Nuru's cache the moment this step completes.

## Project layout

```
engine/          the from-scratch model: tokenizer.py, model.py
data/            synthetic training data + the generator that produces it
app.py           core extraction logic + CLI batch tool
webapp.py        the browser UI (Scan / Review / Automate)
automation.py    webhook / email / archive backends
audit.py         who scanned/sent what, when — metadata only, see /audit
train.py         training loop
evaluate.py      accuracy benchmark harness, see "Extraction accuracy" below
eval/            held-out synthetic evaluation set + the generator that produces it
tests/           pytest suite
legal/           DRAFT ToS/Privacy Policy — not legal advice, see legal/README.md
docs/            reference docs (SOC 2 readiness gap analysis, etc.)
```

## Setup

```
pip install -r requirements.txt
python data/generate_dataset.py   # only needed if you want to regenerate training data
python train.py                   # only needed if engine/weights.npz isn't already there
```

`engine/weights.npz` and `engine/vocab.json` are committed, so a fresh
clone can skip straight to running the app.

### Configure automations (optional)

```
cp nuru.local.env.example nuru.local.env
```

Fill in whichever settings you want — every automation gracefully shows a
"not configured" message if you skip it. See the comments in
`nuru.local.env.example` for what each variable does. This file is
gitignored; it never gets committed.

## Running it

**Local, for yourself:**
```
python webapp.py
```
Opens on http://127.0.0.1:5000.

**A more robust server** (no debug mode, handles concurrent requests better):
```
python -m waitress --host=127.0.0.1 --port=5000 webapp:app
```

If you ever expose Nuru beyond localhost, set `NURU_ACCESS_PASSWORD` first
— every page then requires that password over HTTP Basic Auth. There's no
HTTPS built in, so put it behind a reverse proxy (nginx, Caddy) if it's
going out over a real network.

**Command line, no browser** (scans straight to an Excel file, no review step):
```
python app.py invoice1.pdf invoice2.pdf --output extracted.xlsx
python app.py documents/*.pdf --purge-source
```

## Security

What's in place today:

- **Upload size capped** at 25&nbsp;MB per request (`MAX_CONTENT_LENGTH`).
- **Webhook destinations are checked before sending.** Link-local addresses
  (where cloud metadata endpoints live), multicast, and reserved ranges are
  always blocked; loopback/private addresses are allowed by default (a
  self-hosted receiver on the same machine/LAN is normal for one user) —
  set `NURU_WEBHOOK_PUBLIC_ONLY=true` to also block those. Redirects are
  never followed, closing off the "safe URL that redirects to an unsafe
  one" bypass. This doesn't defend a determined DNS-rebinding attack (the
  hostname resolving safely at check-time but differently at connect-time)
  — that needs a transport that connects to a pinned IP, which isn't
  implemented.
- **Archive destinations are allowlisted**, not free text. The review
  screen only offers a dropdown of `NURU_ARCHIVE_DIR` plus whatever's in
  `NURU_ARCHIVE_ALLOWED_ROOTS` — never an arbitrary path someone typed.
- **CSRF protection** (Flask-WTF) on every state-changing route.
- **Rate limiting** (Flask-Limiter): 60 requests/minute by default, 10/min
  on `/scan`, 20/min on `/automate`. In-memory, so it resets per process
  and doesn't share state across multiple worker processes.
- **Security response headers** on every response: CSP, `X-Frame-Options`,
  `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`, and
  `Strict-Transport-Security` (only takes effect once actually served over
  HTTPS). The CSP still allows `'unsafe-inline'` for styles/scripts, since
  the templates use inline `<style>`/`<script>` blocks rather than
  per-request nonces — a real gap in what it protects against, not a
  solved one.

What's deliberately **not** in place, because it needs a real decision
rather than a default: per-user accounts (there's an audit log now, but
"who" is still an IP address rather than a named person, since auth is
still one shared password or none), TLS termination (put a reverse proxy
in front of it), and a shared rate-limit backend for a multi-process
deployment.

## Compliance & trust

- **Audit trail** (`audit.py`, viewable at `/audit`): every scan and every
  automation run is logged — event, document name, inferred type, action
  and destination, outcome, requester IP, timestamp. Deliberately
  metadata-only; it never stores the extracted field values, so it can't
  become a second permanent copy of the financial data it's supposed to
  be tracking.
- **Bounded retention even for abandoned reviews**: a document that's
  scanned but never carried through to an automation action is
  auto-purged after `NURU_REVIEW_TTL_HOURS` (24h default) — the "deleted
  the moment automation completes" story used to only hold if someone
  actually finished that step.
- **Terms of Service / Privacy Policy**: drafted in [`legal/`](legal/),
  grounded in what the code actually does — **not legal advice, not
  ready to publish**. Read [`legal/README.md`](legal/README.md) first.
- **SOC 2**: [`docs/soc2-readiness.md`](docs/soc2-readiness.md) maps what
  a report would require against what exists today. It's a gap analysis
  to work from, not something code alone gets you — most of it is
  organizational process (written policies, a risk assessment cadence,
  an actual audit engagement), not a repo change.

## Tests

```
pip install pytest reportlab
pytest
```

Covers the tokenizer, the hand-written forward/backward math, entity
decoding and document-type inference, all three automation backends, and
the Flask routes end to end (isolated from any real running instance's
cache and state).

## Extraction accuracy

```
python eval/generate_eval_set.py   # only needed to regenerate the held-out set
python evaluate.py
```

`evaluate.py` measures per-field accuracy, document-type accuracy, and
whether the confidence flag is actually informative (does a wrong answer
get flagged more often than a right one), against `eval/documents/` — a
fixed, held-out set of synthetic PDFs whose vendor/merchant/account names
and phrasing are deliberately absent from `data/generate_dataset.py`'s
training templates, so it measures generalization rather than the model
grading its own homework.

**Current numbers on that set: 100% document-type accuracy, 94% field
accuracy (62/66).** This is **not a real-world accuracy claim** — every
document in `eval/documents/` is still synthetic, and no benchmark against
actual invoices from actual vendors exists. Point `evaluate.py --documents
<dir> --ground-truth <file>` at real documents (same JSON shape as
`eval/ground_truth.json`) the moment any exist.

The remaining known gap, found by running this harness rather than
assumed: a multi-word name (vendor or merchant) with an unfamiliar middle
word occasionally gets tagged with the wrong entity *kind* on its first
token even when later words in the same span are tagged correctly — the
model has no lexical signal for a word it's never seen, only local
context, and that context is sometimes ambiguous between "this starts a
company name" and "this starts a store name." The mismatched tag causes
`decode_entities` to drop or truncate the span rather than guess. A
document-level type-inference fix (checking structurally-unambiguous
fields like Tax before ever trusting an entity-name guess) closed most of
this; what's left needs either a larger/more varied training vocabulary
or character-level features, not another logic patch.

## Known limitations

- **Trained entirely on synthetic data.** See "Extraction accuracy" above
  for the actual measured numbers and how to reproduce them. The review
  step exists specifically to catch what it gets wrong — don't skip it in
  a real workflow.
- **OCR needs Tesseract installed separately.** `pytesseract` (the Python
  client) is in `requirements.txt`, but it calls out to the Tesseract OCR
  engine, which isn't a pip package. Install it from
  [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
  (or `brew install tesseract` / `apt install tesseract-ocr`) for
  scanned/photographed documents to work. Without it, those documents get
  a clear "no readable text found" message instead of a crash.
- **In-memory-plus-a-JSON-file, not a database.** Documents mid-review
  persist across a restart (`webapp.py`'s `_PENDING` is saved to
  `.nuru_cache/pending_state.json`), which is enough for local,
  single-user use. It isn't built for concurrent multi-user traffic.
