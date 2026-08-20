# Nuru

Nuru reads invoices, receipts, and financial statements from PDFs and
turns them into organized records, with a human reviewing and approving
every field before anything leaves the app.

It's built from scratch: the classifier underneath is hand-written NumPy
(embedding lookup, a context window, a ReLU hidden layer, a softmax
output, manually-derived backpropagation), no PyTorch, no TensorFlow, no
pretrained models. Because extraction never leaves this machine, Nuru is
a fit for law firms, healthcare practices, government contractors, and
other regulated-industry teams that are compliance-blocked from sending
financial documents to a cloud AI vendor.

## How it works

**Scan → Review → Automate**, each its own screen:

1. **Scan**: upload one or more PDFs, or a PNG/JPEG/WebP screenshot or
   photo of a document (drag, click to browse, or paste with Ctrl+V). An
   image is wrapped into a one-page PDF automatically, then Nuru extracts
   text (falling back to OCR if a page has no text layer) and classifies
   each token.
2. **Review**: Nuru infers what kind of document it is (Invoice, Receipt,
   or Statement) from which fields it actually found, and shows only the
   fields relevant to that kind, labeled in plain business language. Every
   field is editable: rename it, remove it, or add a custom one.
3. **Automate**: send the approved data to a webhook (Zapier, Make, or
   your own endpoint), email it, archive the source PDF under a clean
   name, save it to Nuru's own built-in ledger (see below), or just
   download it as Excel. "Preview what will be sent" shows the exact
   payload first; no guessing at field names when wiring up a Zap. The
   source file is permanently deleted from Nuru's cache the moment this
   step completes. See [docs/integrations.md](docs/integrations.md) for
   step-by-step Zapier/Make setup.

## Ledger

A native, persistent bookkeeping record, separate from the one-shot
webhook/email/archive/download actions: choosing "Save to ledger" in the
Automate step writes the approved fields into a local SQLite database
(`.nuru_data/ledger.db`) and purges the source PDF exactly like every
other action. Unlike `.nuru_cache/`, this store is never TTL-purged; an
entry stays until someone explicitly deletes it from the `/ledger`
screen. `/ledger` is searchable and filterable (document type, category,
date range, free text) and shows a running total; every saved entry keeps
its full approved field set, even fields the ledger's summary columns
don't recognize by label.

V1 is deliberately scoped down: no multi-user accounts/roles, no true
double-entry accounting or general ledger, no multi-currency conversion,
no payroll, no inventory, no ML-based auto-categorization (categories are
a manual dropdown plus freetext), no bank feed reconciliation, and no
editing a saved entry (delete and re-save from a fresh scan instead). The
schema does carry a little headroom for a possible future Odoo/ERPNext
export connector (currency, tax amount, and a couple of reserved columns)
without any of that integration existing today.

## Trust & custody report

`/trust-report` is a printable page, built entirely from data Nuru
already logs (the audit trail and the ledger), that documents what
actually happened: how many documents were scanned, how they were
automated, and a plain statement that extraction runs locally with no
third-party AI or cloud OCR call involved. It's meant to be something a
compliance reviewer can print or save as a PDF and hand to an auditor. It
deliberately does not claim more than the code actually does: see the
disclaimer at the bottom of the report itself, and
[docs/soc2-readiness.md](docs/soc2-readiness.md) for what a formal
certification would additionally require.

## Project layout

```
engine/          the from-scratch model: tokenizer.py, model.py
data/            synthetic training data + the generator that produces it
app.py           core extraction logic + CLI batch tool
webapp.py        the browser UI (Scan / Review / Automate, Ledger, Audit trail, Trust report)
automation.py    webhook / email / archive backends
ledger.py        the native SQLite bookkeeping record, see "Ledger" above
audit.py         who scanned/sent what, when, metadata only, see /audit
errors.py        structured error logging (+ optional Sentry forwarding)
train.py         training loop
evaluate.py      accuracy benchmark harness, see "Extraction accuracy" below
eval/            held-out synthetic evaluation set + the generator that produces it
tests/           pytest suite
legal/           DRAFT ToS/Privacy Policy, not legal advice, see legal/README.md
docs/            reference docs (integrations guide, SOC 2 readiness gap analysis)
```

`.nuru_cache/` (ephemeral, TTL-purged) and `.nuru_data/` (the ledger,
persistent until explicitly deleted) are both created at runtime and
gitignored.

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

Fill in whichever settings you want; every automation gracefully shows a
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

If you ever expose Nuru beyond localhost, set `NURU_ACCESS_PASSWORD` first;
every page then requires that password over HTTP Basic Auth. There's no
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
  self-hosted receiver on the same machine/LAN is normal for one user);
  set `NURU_WEBHOOK_PUBLIC_ONLY=true` to also block those. Redirects are
  never followed, closing off the "safe URL that redirects to an unsafe
  one" bypass. This doesn't defend a determined DNS-rebinding attack (the
  hostname resolving safely at check-time but differently at connect-time);
  that needs a transport that connects to a pinned IP, which isn't
  implemented.
- **Archive destinations are allowlisted**, not free text. The review
  screen only offers a dropdown of `NURU_ARCHIVE_DIR` plus whatever's in
  `NURU_ARCHIVE_ALLOWED_ROOTS`, never an arbitrary path someone typed.
- **CSRF protection** (Flask-WTF) on every state-changing route.
- **Rate limiting** (Flask-Limiter): 60 requests/minute by default, 10/min
  on `/scan`, 20/min on `/automate`. In-memory, so it resets per process
  and doesn't share state across multiple worker processes.
- **Security response headers** on every response: CSP, `X-Frame-Options`,
  `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`, and
  `Strict-Transport-Security` (only takes effect once actually served over
  HTTPS). The CSP still allows `'unsafe-inline'` for styles/scripts, since
  the templates use inline `<style>`/`<script>` blocks rather than
  per-request nonces; a real gap in what it protects against, not a
  solved one.

What's deliberately **not** in place, because it needs a real decision
rather than a default: per-user accounts (there's an audit log now, but
"who" is still an IP address rather than a named person, since auth is
still one shared password or none), TLS termination (put a reverse proxy
in front of it), and a shared rate-limit backend for a multi-process
deployment.

## Compliance & trust

- **Audit trail** (`audit.py`, viewable at `/audit`): every scan and every
  automation run is logged: event, document name, inferred type, action
  and destination, outcome, requester IP, timestamp. Deliberately
  metadata-only; it never stores the extracted field values, so it can't
  become a second permanent copy of the financial data it's supposed to
  be tracking.
- **Bounded retention even for abandoned reviews**: a document that's
  scanned but never carried through to an automation action is
  auto-purged after `NURU_REVIEW_TTL_HOURS` (24h default); the "deleted
  the moment automation completes" story used to only hold if someone
  actually finished that step.
- **Terms of Service / Privacy Policy**: drafted in [`legal/`](legal/),
  grounded in what the code actually does. **Not legal advice, not
  ready to publish**. Read [`legal/README.md`](legal/README.md) first.
- **SOC 2**: [`docs/soc2-readiness.md`](docs/soc2-readiness.md) maps what
  a report would require against what exists today. It's a gap analysis
  to work from, not something code alone gets you; most of it is
  organizational process (written policies, a risk assessment cadence,
  an actual audit engagement), not a repo change.
- **Trust & custody report** (`/trust-report`): see "Trust & custody
  report" above. Built from the same audit log described in the first
  bullet, plus the ledger's totals; explicitly not an independent
  third-party attestation.

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
get flagged more often than a right one), against `eval/documents/`, a
held-out set of synthetic PDFs (20 hand-authored structural stress tests
plus 51 generated the same combinatorial way training documents are)
whose vendor/merchant/account names and phrasing are deliberately absent
from `data/generate_dataset.py`'s training pools, so it measures
generalization rather than the model grading its own homework.

**Current numbers on that set: 100% document-type accuracy, 98% field
accuracy (255/261).** This is **not a real-world accuracy claim**; every
document in `eval/documents/` is still synthetic, and no benchmark against
actual invoices from actual vendors exists. Point `evaluate.py --documents
<dir> --ground-truth <file>` at real documents (same JSON shape as
`eval/ground_truth.json`) the moment any exist.

**How the model stays capable without any pretrained weights**:
`engine/tokenizer.py` gives every word, including one it's never seen, a
representation built from hashed character n-grams of its own spelling
(fastText's *hashing trick*, trained from scratch, nothing pretrained),
concatenated with its word embedding in `engine/model.py`. This directly
targets the failure mode this project found by running the eval harness
rather than assuming: an unfamiliar multi-word name (vendor or merchant)
could get tagged with the wrong entity *kind*, or have its later words
dropped, because the model had no lexical signal for a word it had never
seen. `data/generate_dataset.py` also grew from ~45 hand-typed vendor
names to combinatorially generated ones (name-fragment pools split
85/15 so training and eval structurally never share an identifying name
fragment) with far more phrasing variety and realistic non-entity
content (addresses, item lines, terms boilerplate) spliced in at
randomized positions, so entities are learned embedded in surrounding
text rather than always in the same relative spot.

The remaining known gap: a long (3-4 word) span, especially an
ampersand-style firm name ("Jarvinen & Bramblewood Tavern"), still
occasionally loses its last word or gets dropped entirely. The model
only ever sees a fixed 5-token local window (2 tokens either side); a
span running the full length of that window has little room left to
confirm it should keep tagging. Widening the window, or moving past a
fixed-window feed-forward tagger to something with real sequence memory,
would target this directly, but was deliberately deferred rather than
bundled into the same change as the character-feature/data work above,
to keep each change's effect on accuracy separately measurable.

## Operations

- **Error logging** (`errors.py`): every caught failure (a page that
  won't parse, a webhook that won't send, an unexpected exception in a
  route) goes to `.nuru_cache/error.log` (rotated at 2&nbsp;MB, 3
  backups kept), not just wherever stdout happens to be pointed. Set
  `SENTRY_DSN` to also forward exceptions to Sentry; without it, logging
  stays local-only.
- **Health check**: `GET /healthz` returns `{"status": "ok"}` (200) or
  `{"status": "unhealthy"}` (503) based on whether the model actually
  loaded. Exempt from the access password and rate limiting, since a load
  balancer or uptime monitor has no way to carry credentials.
- **Docker**: `docker build -t nuru .`. The image installs Tesseract, so
  OCR works out of the box in a container even though it's optional
  locally. Runs as a non-root user, has a `HEALTHCHECK` wired to
  `/healthz`, and serves via waitress rather than Flask's dev server. Pass
  real config with `docker run --env-file nuru.local.env ...` or individual
  `-e` flags. Verified this session: built, ran, and pushed a real
  document through it end to end, including a real OCR pass on a
  scanned-style PDF.
- **CI** (`.github/workflows/tests.yml`): runs the full pytest suite on
  every push and pull request to `main`.

## Known limitations

- **Trained entirely on synthetic data.** See "Extraction accuracy" above
  for the actual measured numbers and how to reproduce them. The review
  step exists specifically to catch what it gets wrong; don't skip it in
  a real workflow.
- **OCR needs Tesseract installed separately outside Docker.** `pytesseract`
  (the Python client) is in `requirements.txt`, but it calls out to the
  Tesseract OCR
  engine, which isn't a pip package. Install it from
  [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
  (or `brew install tesseract` / `apt install tesseract-ocr`) for
  scanned/photographed documents to work. Without it, those documents get
  a clear "no readable text found" message instead of a crash.
- **In-memory-plus-a-JSON-file, not a database, for review state.**
  Documents mid-review persist across a restart (`webapp.py`'s `_PENDING`
  is saved to `.nuru_cache/pending_state.json`), which is enough for
  local, single-user use. It isn't built for concurrent multi-user
  traffic. (The ledger itself, once a document is saved there, is a real
  SQLite database; this limitation is about the in-progress review queue
  only.)
- **The ledger's summary columns are best-effort.** `amount`,
  `document_date`, and `counterparty` are populated by matching the
  approved field labels against known names (Supplier Name/Merchant/
  Account Name, Total Amount Due/Amount Paid/Closing Balance, etc.). A
  heavily renamed or custom field won't populate them, though the full
  approved field set is always preserved and viewable per entry.
