"""Downloads a real-world receipt benchmark: SROIE (ICDAR 2019 Robust
Reading Challenge on Scanned Receipts OCR and Information Extraction),
via the MIT-licensed mirror at github.com/zzzDavid/ICDAR-2019-SROIE.

Every eval set elsewhere in this project is synthetic; this is the one
genuine real-world check Nuru has, receipts only (SROIE has no invoice or
statement equivalent with comparable field coverage and a clear license;
see README's Extraction accuracy section for why that's a documented gap
rather than a filled one). Images are real photographed/scanned Malaysian
retail receipts, run through Nuru's actual pipeline unmodified (image ->
image_to_pdf -> OCR fallback -> the trained classifier), not through
SROIE's own text-localization annotations, since that would benchmark a
different OCR step than the one Nuru ships with.

Downloaded images and the converted ground truth are gitignored (not
committed): re-run this script to reproduce them. Usage:

    python eval/real_data/prepare_sroie.py                # 150-receipt sample
    python eval/real_data/prepare_sroie.py --count 626     # the full public set

Attribution (SROIE's license requires it): Task 3 data from the ICDAR
2019 Robust Reading Challenge on Scanned Receipts OCR and Information
Extraction (SROIE), mirrored at github.com/zzzDavid/ICDAR-2019-SROIE
under the MIT License.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app import image_to_pdf

REAL_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(REAL_DATA_DIR, "sroie", "documents")
GROUND_TRUTH_PATH = os.path.join(REAL_DATA_DIR, "sroie", "ground_truth.json")

_RAW_BASE = "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data"
_TOTAL_AVAILABLE = 626

# SROIE's four annotated fields; only company/date/total map onto anything
# Nuru's Receipt schema extracts. address has no Nuru equivalent, so it's
# dropped rather than forced into a field that doesn't exist.
_FIELD_MAP = {"company": "Merchant", "date": "Purchase Date", "total": "Amount Paid"}


def _fetch(path):
    url = f"{_RAW_BASE}/{path}"
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=150,
                         help=f"How many receipts to download (max {_TOTAL_AVAILABLE}). "
                              "150 is a solid sample without a very long download.")
    args = parser.parse_args()
    count = min(args.count, _TOTAL_AVAILABLE)

    os.makedirs(DOCS_DIR, exist_ok=True)
    ground_truth = {}
    fetched = 0
    skipped = 0

    for i in range(count):
        stem = f"{i:03d}"
        pdf_name = f"{stem}.pdf"
        raw_img_path = os.path.join(DOCS_DIR, f"{stem}.jpg")
        pdf_path = os.path.join(DOCS_DIR, pdf_name)

        try:
            key_bytes = _fetch(f"key/{stem}.json")
        except urllib.error.HTTPError:
            skipped += 1
            continue

        key = json.loads(key_bytes)
        fields = {nuru_label: key[sroie_key] for sroie_key, nuru_label in _FIELD_MAP.items() if sroie_key in key}
        if not fields:
            skipped += 1
            continue

        try:
            img_bytes = _fetch(f"img/{stem}.jpg")
        except urllib.error.HTTPError:
            skipped += 1
            continue

        # evaluate.py's harness (like every other eval set in this project)
        # expects one PDF per document; converting here, once, at download
        # time keeps evaluate.py itself untouched and matches exactly what
        # webapp.py's real upload path does for an image (image_to_pdf then
        # the normal PDF pipeline), so this benchmarks Nuru's real OCR
        # fallback, not SROIE's own text-localization annotations.
        with open(raw_img_path, "wb") as f:
            f.write(img_bytes)
        try:
            image_to_pdf(raw_img_path, pdf_path)
        except Exception as exc:
            print(f"  skipping {stem}: could not convert image to PDF ({exc})")
            skipped += 1
            continue
        finally:
            os.remove(raw_img_path)

        ground_truth[pdf_name] = {"type": "Receipt", "fields": fields}
        fetched += 1
        if fetched % 25 == 0:
            print(f"  {fetched}/{count} downloaded...")

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"\nFetched {fetched} real receipts ({skipped} skipped) to {DOCS_DIR}")
    print(f"Wrote ground truth to {GROUND_TRUTH_PATH}")
    print("\nRun the benchmark with:")
    print(f"  python evaluate.py --documents {DOCS_DIR} --ground-truth {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
