"""Builds real, weakly-labeled training data from SROIE receipts distinct
from the ones used as the real-world eval benchmark (eval/real_data/),
so training never sees the same documents used to measure it.

For each receipt: runs it through Nuru's actual pipeline (image_to_pdf,
then the same extract_text/OCR fallback used at real inference time,
not SROIE's own text annotations), then fuzzy-aligns SROIE's ground
truth (company/date/total) against the resulting noisy tokens via
data/align_real_labels.py to produce BIO labels. A field that can't be
confidently aligned is simply left unlabeled for that document (see
align_real_labels.py's docstring for why that's the safe default), not
guessed at.

Output is data/real_data.csv, same doc_id,token,label shape as
data/generate_dataset.py's synthetic output, with string doc_ids
prefixed "real_" so they can never collide with the synthetic
generator's integer ids. Gitignored, not committed: re-run this script
to reproduce it, same as eval/real_data/prepare_sroie.py.

Usage:
    python data/prepare_real_training_data.py                  # 400-receipt sample
    python data/prepare_real_training_data.py --start 150 --count 400
"""

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import extract_text, image_to_pdf
from data.align_real_labels import build_bio_labels, find_best_text_span, find_date_span, find_money_span
from engine.tokenizer import tokenize

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_DATA_DIR = os.path.join(DATA_DIR, "real_data")
OUT_PATH = os.path.join(DATA_DIR, "real_data.csv")

_RAW_BASE = "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data"
_TOTAL_AVAILABLE = 626
_FIELD_MAP = {"company": "Merchant", "date": "Purchase Date", "total": "Amount Paid"}
_LABEL_TO_ENTITY = {"Merchant": "Merchant", "Purchase Date": "Date", "Amount Paid": "Total"}


def _fetch(path):
    url = f"{_RAW_BASE}/{path}"
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read()


def _align_document(tokens, ground_truth_fields):
    """Returns (labels, matched_entities) for one document's tokens."""
    spans = {}
    if "Merchant" in ground_truth_fields:
        span = find_best_text_span(tokens, ground_truth_fields["Merchant"])
        spans["Merchant"] = (span[0], span[1]) if span else None
    if "Purchase Date" in ground_truth_fields:
        spans["Date"] = find_date_span(tokens, ground_truth_fields["Purchase Date"])
    if "Amount Paid" in ground_truth_fields:
        spans["Total"] = find_money_span(tokens, ground_truth_fields["Amount Paid"])

    labels = build_bio_labels(tokens, spans)
    matched = {k for k, v in spans.items() if v is not None}
    return labels, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=150,
                         help="First SROIE index to use (default 150, right after the 150 reserved for eval).")
    parser.add_argument("--count", type=int, default=400)
    args = parser.parse_args()
    end = min(args.start + args.count, _TOTAL_AVAILABLE)

    os.makedirs(REAL_DATA_DIR, exist_ok=True)
    rows = []
    doc_count = 0
    match_totals = {"Merchant": 0, "Date": 0, "Total": 0}
    skipped = 0

    for i in range(args.start, end):
        stem = f"{i:03d}"
        raw_img_path = os.path.join(REAL_DATA_DIR, f"{stem}.jpg")
        pdf_path = os.path.join(REAL_DATA_DIR, f"{stem}.pdf")

        try:
            key_bytes = _fetch(f"key/{stem}.json")
            img_bytes = _fetch(f"img/{stem}.jpg")
        except urllib.error.HTTPError:
            skipped += 1
            continue

        key = json.loads(key_bytes)
        fields = {nuru_label: key[sroie_key] for sroie_key, nuru_label in _FIELD_MAP.items() if sroie_key in key}
        if not fields:
            skipped += 1
            continue

        with open(raw_img_path, "wb") as f:
            f.write(img_bytes)
        try:
            image_to_pdf(raw_img_path, pdf_path)
        except Exception:
            skipped += 1
            os.remove(raw_img_path)
            continue
        os.remove(raw_img_path)

        text, failed_pages, used_ocr = extract_text(pdf_path)
        os.remove(pdf_path)  # only the CSV row is kept; the intermediate PDF isn't needed after this
        tokens = tokenize(text)
        if not tokens:
            skipped += 1
            continue

        labels, matched = _align_document(tokens, fields)
        if not matched:
            skipped += 1  # nothing usable was aligned at all; contributes no training signal
            continue

        for entity in matched:
            match_totals[entity] += 1
        doc_id = f"real_{stem}"
        for tok, lab in zip(tokens, labels):
            rows.append((doc_id, tok, lab))
        doc_count += 1
        if doc_count % 50 == 0:
            print(f"  {doc_count} documents processed...")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "token", "label"])
        writer.writerows(rows)

    print(f"\n{doc_count} real documents written ({skipped} skipped) to {OUT_PATH}")
    print("Fields successfully aligned:")
    for entity, count in match_totals.items():
        print(f"  {entity}: {count}/{doc_count}")


if __name__ == "__main__":
    main()
