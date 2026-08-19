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
train.py         training loop
tests/           pytest suite
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

## Tests

```
pip install pytest reportlab
pytest
```

Covers the tokenizer, the hand-written forward/backward math, entity
decoding and document-type inference, all three automation backends, and
the Flask routes end to end (isolated from any real running instance's
cache and state).

## Known limitations

- **Trained entirely on synthetic data.** It generalizes well to phrasing
  and layouts it hasn't seen exactly (see the model's context-window
  design in `engine/model.py`), but real-world documents vary more than
  any synthetic generator can fully anticipate. The review step exists
  specifically to catch what it gets wrong — don't skip it in a real
  workflow.
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
