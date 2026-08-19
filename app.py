"""Nuru: user-facing interface. PDFs in (invoices, receipts, or financial
statements), a single Excel workbook out.

Runs the trained pure-NumPy token classifier over the text extracted from
each PDF, decodes the BIO tag sequence into whichever fields are relevant
to that document's kind, and writes one row per document to an .xlsx file.

Different document kinds surface different fields: an invoice reports a
supplier and a tax value, a receipt reports a merchant and a payment
method, a statement reports an account and a closing balance. Which kind a
given file is gets inferred from which entities the model actually found
in it (see infer_document_type below) rather than being fixed in advance,
so the field set genuinely depends on what was uploaded.

Real documents are messy: a page can fail to parse, an amount can hit an
untrained corner of the model. Two things follow from that:
  - Extraction is wrapped in per-page and per-document error boundaries, so
    one bad page or one unreadable PDF never aborts the rest of a bulk run.
  - Every predicted field carries its softmax confidence; anything under
    the threshold is flagged with "Needs Review" and highlighted orange in
    the workbook so a human double-checks it instead of trusting it blind.

Usage:
    python app.py invoice1.pdf receipt2.pdf --output extracted.xlsx
    python app.py documents/*.pdf
    python app.py documents/*.pdf --purge-source   # delete PDFs once captured in the .xlsx
"""

import argparse
import glob
import os

import pdfplumber
from openpyxl import Workbook
from openpyxl.styles import PatternFill

from engine.model import ID2LABEL, TokenClassifier, build_context_windows
from engine.tokenizer import Vocabulary, tokenize

ENGINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine")
WEIGHTS_PATH = os.path.join(ENGINE_DIR, "weights.npz")
VOCAB_PATH = os.path.join(ENGINE_DIR, "vocab.json")

DEFAULT_CONFIDENCE_THRESHOLD = 0.70

# Underlying entity -> the professional term shown to the user, per document
# kind. Same extraction model, different vocabulary depending on what was
# actually uploaded: a "Total" reads as "Total Amount Due" on an invoice but
# "Amount Paid" on a receipt.
TYPE_FIELD_CONFIG = {
    "Invoice": [
        ("Vendor", "Supplier Name"),
        ("Date", "Transaction Date"),
        ("Total", "Total Amount Due"),
        ("Tax", "Tax Value (VAT/GST)"),
    ],
    "Receipt": [
        ("Merchant", "Merchant"),
        ("Date", "Purchase Date"),
        ("Total", "Amount Paid"),
        ("PaymentMethod", "Payment Method"),
    ],
    "Statement": [
        ("Account", "Account Name"),
        ("Period", "Statement Period"),
        ("Balance", "Closing Balance"),
    ],
}

# Which entities, if the model finds them at all, are strong evidence of
# which document kind. Checked in order; first match wins.
_TYPE_INDICATORS = [
    ("Receipt", {"Merchant", "PaymentMethod"}),
    ("Statement", {"Account", "Period", "Balance"}),
    ("Invoice", {"Vendor", "Tax"}),
]

MAX_FIELDS_PER_TYPE = max(len(cfg) for cfg in TYPE_FIELD_CONFIG.values())

LOW_CONFIDENCE_FILL = PatternFill(start_color="FFCC80", end_color="FFCC80", fill_type="solid")
ERROR_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")


def infer_document_type(found_entities):
    """Pick a document kind from the set of entity keys the model actually
    predicted somewhere in the document. Falls back to Invoice, the most
    common case, when nothing distinctive fired."""
    for type_name, indicators in _TYPE_INDICATORS:
        if found_entities & indicators:
            return type_name
    return "Invoice"


def extract_text(pdf_path):
    """Extract text page by page.

    A page that fails to parse (corrupt content stream, malformed table,
    unsupported encoding) is skipped rather than raising, so it doesn't take
    the rest of the document down with it. Returns (text, failed_page_numbers).
    """
    text_parts = []
    failed_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                failed_pages.append(page_num)
                print(f"  warning: page {page_num} failed to parse ({exc}); skipping")
                continue
            text_parts.append(page_text)
    return "\n".join(text_parts), failed_pages


def decode_entities(tokens, pred_labels, confidences):
    """Turn a per-token BIO prediction into {entity: [values]} spans, plus
    the matching per-span confidence (the weakest token in a span, since a
    multi-token span is only as trustworthy as its shakiest token).

    Generic over whatever entity types the label set defines (B-Vendor,
    B-Merchant, B-Account, ...) rather than hardcoding one field, so it
    doesn't need to change as document kinds are added."""
    fields = {}
    confs = {}
    span_entity = None
    span_tokens, span_confs = [], []

    def flush_span():
        if span_entity and span_tokens:
            fields.setdefault(span_entity, []).append(" ".join(span_tokens))
            confs.setdefault(span_entity, []).append(min(span_confs))
        span_tokens.clear()
        span_confs.clear()

    for tok, label, conf in zip(tokens, pred_labels, confidences):
        if label.startswith("B-"):
            flush_span()
            span_entity = label[2:]
            span_tokens.append(tok)
            span_confs.append(conf)
        elif label.startswith("I-") and label[2:] == span_entity:
            span_tokens.append(tok)
            span_confs.append(conf)
        else:
            flush_span()
            span_entity = None
    flush_span()
    return fields, confs


def _empty_result(note):
    """A flagged, all-empty row. The reason for the failure is logged to the
    console for whoever is running this; the note shown to the end user
    stays in plain language, with no exception text or internals in it."""
    return {
        "Type": None,
        "Fields": [],
        "Error": True,
        "Notes": note,
        "NeedsReview": True,
    }


def _friendly_notes(low_conf_fields, missing_fields, failed_pages):
    parts = []
    if missing_fields:
        parts.append("Couldn't find: " + ", ".join(missing_fields))
    if low_conf_fields:
        parts.append("Worth a second look: " + ", ".join(low_conf_fields))
    if failed_pages:
        word = "page" if len(failed_pages) == 1 else "pages"
        parts.append(f"Part of this document (the {word} in question) wasn't clear enough to read")
    return " · ".join(parts)


def process_invoice(model, vocab, pdf_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """Never raises for document-level problems; a failure becomes a
    flagged row (Error set, NeedsReview true) instead of an exception, so a
    bulk run keeps going through the rest of the batch."""
    try:
        text, failed_pages = extract_text(pdf_path)
    except Exception as exc:
        print(f"  error opening {pdf_path}: {exc}")
        return _empty_result("This file couldn't be opened. It may be damaged or not a valid PDF.")

    tokens = tokenize(text)
    if not tokens:
        note = "No readable text was found in this document."
        if failed_pages:
            note = "This document wasn't clear enough to read."
        return _empty_result(note)

    try:
        ids = vocab.encode(tokens)
        window_ids = build_context_windows(ids, model.window)
        pred_ids, confidences = model.predict_with_confidence(window_ids)
        pred_labels = [ID2LABEL[i] for i in pred_ids]
        entity_values, entity_confs = decode_entities(tokens, pred_labels, confidences)
    except Exception as exc:
        print(f"  error reading {pdf_path}: {exc}")
        return _empty_result("Something went wrong while reading this document. Please try again.")

    doc_type = infer_document_type(set(entity_values.keys()))
    field_config = TYPE_FIELD_CONFIG[doc_type]

    fields = []
    low_conf_labels = []
    missing_labels = []
    for entity_key, display_label in field_config:
        values = entity_values.get(entity_key) or []
        confs = entity_confs.get(entity_key) or []
        value = values[0] if values else ""
        conf = confs[0] if confs else None
        fields.append({"key": entity_key, "label": display_label, "value": value, "confidence": conf})
        if value and conf is not None and conf < confidence_threshold:
            low_conf_labels.append(f"{display_label} (we're {conf:.0%} sure)")
        elif not value:
            missing_labels.append(display_label)

    return {
        "Type": doc_type,
        "Fields": fields,
        "Error": False,
        "Notes": _friendly_notes(low_conf_labels, missing_labels, failed_pages),
        "NeedsReview": bool(low_conf_labels) or bool(failed_pages) or bool(missing_labels),
    }


def write_excel(rows, output_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """Each row's own inferred type picks its own field labels, so a batch
    mixing invoices, receipts, and statements doesn't force one header row
    onto all of them. Column layout is generic slots (Field 1/Value 1, ...),
    sized to whatever the widest row in this call actually has (a
    user-reviewed row can carry more custom fields than any built-in type
    schema), since a flat sheet can't otherwise represent rows with
    genuinely different schemas."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Nuru"
    max_fields = max((len(row.get("Fields", [])) for row in rows), default=0)
    max_fields = max(max_fields, 1)
    headers = ["Document", "Type"]
    for i in range(1, max_fields + 1):
        headers += [f"Field {i}", f"Value {i}"]
    headers += ["Please Verify", "Details"]
    ws.append(headers)

    for row in rows:
        fields = row.get("Fields", [])
        line = [row["File"], row.get("Type") or ""]
        for i in range(max_fields):
            if i < len(fields):
                line += [fields[i]["label"], fields[i]["value"]]
            else:
                line += ["", ""]
        line += [
            "Please double-check" if row.get("NeedsReview") else "Looks good",
            row.get("Notes", ""),
        ]
        ws.append(line)
        r = ws.max_row

        if row.get("Error"):
            for col in range(1, len(headers) + 1):
                ws.cell(row=r, column=col).fill = ERROR_FILL
            continue
        for i, field in enumerate(fields):
            value_col = 4 + i * 2  # Document=1, Type=2, Field1=3, Value1=4, Field2=5, Value2=6, ...
            conf = field.get("confidence")
            low_confidence = field.get("value") and conf is not None and conf < confidence_threshold
            missing = not field.get("value")
            if low_confidence or missing:
                ws.cell(row=r, column=value_col).fill = LOW_CONFIDENCE_FILL
        if row.get("NeedsReview"):
            ws.cell(row=r, column=len(headers) - 1).fill = LOW_CONFIDENCE_FILL

    for col_cells in ws.columns:
        width = max(len(str(cell.value)) for cell in col_cells if cell.value is not None) + 2
        ws.column_dimensions[col_cells[0].column_letter].width = max(width, 10)
    wb.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Nuru: read invoices, get organized records back.")
    parser.add_argument("pdfs", nargs="+", help="PDF file paths or glob patterns.")
    parser.add_argument("--output", default="nuru-invoices.xlsx")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                         help="Softmax probability below which a field is flagged for review (default 0.70).")
    parser.add_argument("--purge-source", action="store_true",
                         help="Permanently delete each source PDF once it has been successfully "
                              "captured in the output Excel file. Off by default, since these are your "
                              "original files on disk, not a server-side cache; only enable this if "
                              "that is genuinely what you want.")
    args = parser.parse_args()

    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(VOCAB_PATH):
        raise SystemExit(
            f"No trained model found at {WEIGHTS_PATH} / {VOCAB_PATH}.\n"
            f"Run `python train.py` first."
        )

    vocab = Vocabulary.load(VOCAB_PATH)
    model = TokenClassifier.load_weights(WEIGHTS_PATH)

    pdf_paths = []
    for pattern in args.pdfs:
        matches = glob.glob(pattern)
        pdf_paths.extend(matches if matches else [pattern])

    rows = []
    succeeded_paths = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f"skip (not found): {pdf_path}")
            continue
        print(f"processing: {pdf_path}")
        try:
            fields = process_invoice(model, vocab, pdf_path, args.confidence_threshold)
        except Exception as exc:
            # Belt-and-suspenders: process_invoice already catches document-level
            # failures, but nothing should be able to take the whole batch down.
            print(f"  error: unexpected failure on {pdf_path}: {exc}")
            fields = _empty_result("Something went wrong while reading this document. Please try again.")
        fields["File"] = os.path.basename(pdf_path)
        rows.append(fields)
        if not fields.get("Error"):
            succeeded_paths.append(pdf_path)

    if not rows:
        raise SystemExit("No PDFs were processed.")

    write_excel(rows, args.output, args.confidence_threshold)
    print(f"Wrote {len(rows)} row(s) to {args.output}")

    if args.purge_source:
        for pdf_path in succeeded_paths:
            try:
                os.remove(pdf_path)
                print(f"purged source: {pdf_path}")
            except OSError as exc:
                print(f"  warning: could not purge {pdf_path}: {exc}")


if __name__ == "__main__":
    main()
