"""Builds a small, fixed, held-out evaluation set: real PDF files (via
reportlab) with hand-written ground truth, deliberately using names and
phrasing that never appear in data/generate_dataset.py's training
templates. This is still synthetic, so it cannot substitute for real-world
invoices, but it's a genuine generalization test rather than the model
grading its own homework on data shaped exactly like what it trained on.

Two kinds of cases live in this file:
  - CASES: 20 hand-authored documents, including deliberate structural
    stress tests (reversed field order, minimal documents, two cases
    built specifically to probe character-level subword generalization).
  - _generated_cases(): documents built the same combinatorial way
    data/generate_dataset.py builds training documents, but drawing only
    from each pool's *_EVAL half (data/generate_dataset.py's fixed,
    seeded 85/15 split), so "never seen in training" is a structural
    guarantee of the split rather than something to remember by hand.

Run once to (re)generate eval/documents/*.pdf and eval/ground_truth.json.
Both are checked into the repo so `evaluate.py` runs reproducibly without
needing this script re-run.
"""

import json
import os
import sys

import numpy as np
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_dataset import (
    BUSINESS_NOUNS_EVAL, DESCRIPTOR_WORDS_EVAL, FIRST_NAMES_EVAL, LAST_NAMES_EVAL,
    MERCHANT_DESCRIPTORS_EVAL, MERCHANT_NOUNS_EVAL, PAYMENT_METHODS, VENDOR_SUFFIXES_EVAL,
    generate_account_holder_name, generate_merchant_name, generate_vendor_name,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(EVAL_DIR, "documents")
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.json")

_GEN_SEED = 99  # independent of data/generate_dataset.py's RNG_SEED and the *_TRAIN/*_EVAL split seed


def _make_pdf(filename, lines):
    path = os.path.join(DOCS_DIR, filename)
    c = canvas.Canvas(path)
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()


# Every name below is deliberately absent from data/generate_dataset.py's
# training-side (*_TRAIN) name pools, so this measures generalization, not
# recall of the training distribution.
CASES = [
    # --- Invoices: varied header/date/total phrasing, unseen vendor names ---
    # inv_01 deliberately puts Total before Tax, the reverse of every
    # training example (which always goes Subtotal -> Tax -> Total): an
    # explicit order-robustness stress test, not representative of the
    # other cases below, which use the conventional order.
    ("inv_01_reversed_tax_order.pdf", [
        "Brightwater Consulting Group",
        "Invoice Date: 2026-02-14",
        "Total Due: $2,410.00",
        "Tax (8.25%): $184.05",
    ], "Invoice", {
        "Supplier Name": "Brightwater Consulting Group",
        "Transaction Date": "2026-02-14",
        "Total Amount Due": "$2,410.00",
        "Tax Value (VAT/GST)": "$184.05",
    }),
    ("inv_02_from_prefix.pdf", [
        "From: Palisade Marine Supply",
        "Date: 09/03/2026",
        "Sales Tax (6%): $52.53",
        "Amount Due: $875.50",
    ], "Invoice", {
        "Supplier Name": "Palisade Marine Supply",
        "Transaction Date": "09/03/2026",
        "Total Amount Due": "$875.50",
        "Tax Value (VAT/GST)": "$52.53",
    }),
    ("inv_03_vendor_label.pdf", [
        "Vendor: Kestrel Data Systems",
        "Issued: 2026-11-30",
        "VAT (7.5%): $998.90",
        "Grand Total: $14,320.75",
    ], "Invoice", {
        "Supplier Name": "Kestrel Data Systems",
        "Transaction Date": "2026-11-30",
        "Total Amount Due": "$14,320.75",
        "Tax Value (VAT/GST)": "$998.90",
    }),
    ("inv_04_billed_by.pdf", [
        "Billed By: Northgate Fabrication Co",
        "Invoice Date: 05/22/2026",
        "Tax (5%): $26.50",
        "Total: $530.00",
    ], "Invoice", {
        "Supplier Name": "Northgate Fabrication Co",
        "Transaction Date": "05/22/2026",
        "Total Amount Due": "$530.00",
        "Tax Value (VAT/GST)": "$26.50",
    }),
    ("inv_05_minimal.pdf", [
        "Sable Ridge Vineyards",
        "Date: 2026-07-04",
        "Tax (8%): $4.53",
        "Total Due: $61.20",
    ], "Invoice", {
        "Supplier Name": "Sable Ridge Vineyards",
        "Transaction Date": "2026-07-04",
        "Total Amount Due": "$61.20",
        "Tax Value (VAT/GST)": "$4.53",
    }),
    ("inv_06_full_boilerplate.pdf", [
        "Harrow & Finch Legal Services",
        "Invoice Number: LG-40218",
        "Invoice Date: 2026-03-19",
        "Bill To: Customer Account # 9931",
        "Subtotal: $6,000.00",
        "VAT (9%): $540.00",
        "Grand Total: $6,540.00",
        "Questions about this invoice? Contact billing.",
    ], "Invoice", {
        "Supplier Name": "Harrow & Finch Legal Services",
        "Transaction Date": "2026-03-19",
        "Total Amount Due": "$6,540.00",
        "Tax Value (VAT/GST)": "$540.00",
    }),

    # --- Receipts: unseen merchants, varied phrasing ---
    ("rec_01_standard.pdf", [
        "Copperline Hardware",
        "Purchase Date: 2026-01-09",
        "Amount Paid: $47.88",
        "Payment Method: Debit",
    ], "Receipt", {
        "Merchant": "Copperline Hardware",
        "Purchase Date": "2026-01-09",
        "Amount Paid": "$47.88",
        "Payment Method": "Debit",
    }),
    ("rec_02_sold_by.pdf", [
        "Sold By: Thistle & Bramble Florist",
        "Date: 04/18/2026",
        "Total: $32.00",
        "Paid via: Cash",
    ], "Receipt", {
        "Merchant": "Thistle & Bramble Florist",
        "Purchase Date": "04/18/2026",
        "Amount Paid": "$32.00",
        "Payment Method": "Cash",
    }),
    ("rec_03_merchant_label.pdf", [
        "Merchant: Golden Hour Photography Studio",
        "Transaction Date: 2026-06-27",
        "Amount: $210.00",
        "Tender: Amex",
    ], "Receipt", {
        "Merchant": "Golden Hour Photography Studio",
        "Purchase Date": "2026-06-27",
        "Amount Paid": "$210.00",
        "Payment Method": "Amex",
    }),
    ("rec_04_minimal.pdf", [
        "Driftwood Surf Rentals",
        "Date: 2026-08-02",
        "Amount Paid: $18.50",
        "Payment Method: Visa",
    ], "Receipt", {
        "Merchant": "Driftwood Surf Rentals",
        "Purchase Date": "2026-08-02",
        "Amount Paid": "$18.50",
        "Payment Method": "Visa",
    }),
    ("rec_05_with_items.pdf", [
        "Anchor Point Deli",
        "Purchase Date: 12/24/2026",
        "Item Sundry goods x 3",
        "Total: $15.75",
        "Paid via: PayPal",
        "Come back soon!",
    ], "Receipt", {
        "Merchant": "Anchor Point Deli",
        "Purchase Date": "12/24/2026",
        "Amount Paid": "$15.75",
        "Payment Method": "PayPal",
    }),
    ("rec_06_full_boilerplate.pdf", [
        "Sold By: Marrow Street Butchers",
        "Purchase Date: 2026-09-15",
        "Item Sundry goods x 1",
        "Amount Paid: $88.40",
        "Payment Method: Mastercard",
        "Thank you for shopping with us.",
    ], "Receipt", {
        "Merchant": "Marrow Street Butchers",
        "Purchase Date": "2026-09-15",
        "Amount Paid": "$88.40",
        "Payment Method": "Mastercard",
    }),

    # --- Statements: unseen account holders, varied phrasing ---
    ("stmt_01_standard.pdf", [
        "Tobias Ferreira-Lindqvist",
        "Statement Period: April 2026",
        "Closing Balance: $9,214.60",
    ], "Statement", {
        "Account Name": "Tobias Ferreira-Lindqvist",
        "Statement Period": "April 2026",
        "Closing Balance": "$9,214.60",
    }),
    ("stmt_02_holder_label.pdf", [
        "Account Holder: Wren Okonkwo-Baptiste",
        "Billing Period: September 2026",
        "Balance: $1,042.15",
    ], "Statement", {
        "Account Name": "Wren Okonkwo-Baptiste",
        "Statement Period": "September 2026",
        "Closing Balance": "$1,042.15",
    }),
    ("stmt_03_prepared_for.pdf", [
        "Prepared For: Dashiell Vantongeren",
        "Period: January 2026",
        "Ending Balance: $18,730.00",
    ], "Statement", {
        "Account Name": "Dashiell Vantongeren",
        "Statement Period": "January 2026",
        "Closing Balance": "$18,730.00",
    }),
    ("stmt_04_minimal.pdf", [
        "Ines Castellano-Reyes",
        "Statement Period: June 2026",
        "Closing Balance: $327.90",
    ], "Statement", {
        "Account Name": "Ines Castellano-Reyes",
        "Statement Period": "June 2026",
        "Closing Balance": "$327.90",
    }),
    ("stmt_05_with_summary.pdf", [
        "Odalys Marchetti-Whitfield",
        "Account Summary",
        "Statement Period: November 2026",
        "Opening Balance: $2,100.00",
        "Closing Balance: $2,955.40",
    ], "Statement", {
        "Account Name": "Odalys Marchetti-Whitfield",
        "Statement Period": "November 2026",
        "Closing Balance": "$2,955.40",
    }),
    ("stmt_06_full_boilerplate.pdf", [
        "Account Holder: Percival Nakashima-Ortiz",
        "Account Summary",
        "Statement Period: December 2026",
        "Opening Balance: $500.00",
        "Closing Balance: $1,275.00",
        "Please review your transactions and report any discrepancies.",
    ], "Statement", {
        "Account Name": "Percival Nakashima-Ortiz",
        "Statement Period": "December 2026",
        "Closing Balance": "$1,275.00",
    }),

    # --- Char-feature stress tests: two OOV vendor names, one that shares
    # spelling with a training-vocabulary word (subword overlap should
    # help), one that shares nothing (the pure-lexical floor). Neither
    # "Ironhaven"/"Zylquorth" is a literal training name; "Iron"/"Ironwood"
    # are real entries in data/generate_dataset.py's training-side
    # descriptor pool, so the first case has genuine, checkable overlap. ---
    ("inv_char_overlap_with_training_subwords.pdf", [
        "Ironhaven Fabrication Partners",
        "Invoice Date: 2026-05-11",
        "Tax (7%): $140.70",
        "Total Due: $2,150.70",
    ], "Invoice", {
        "Supplier Name": "Ironhaven Fabrication Partners",
        "Transaction Date": "2026-05-11",
        "Total Amount Due": "$2,150.70",
        "Tax Value (VAT/GST)": "$140.70",
    }),
    ("inv_char_no_training_subword_overlap.pdf", [
        "Zylquorth Vantrex Holdings",
        "Invoice Date: 2026-05-12",
        "Tax (7%): $98.00",
        "Total Due: $1,498.00",
    ], "Invoice", {
        "Supplier Name": "Zylquorth Vantrex Holdings",
        "Transaction Date": "2026-05-12",
        "Total Amount Due": "$1,498.00",
        "Tax Value (VAT/GST)": "$98.00",
    }),
]


def _pick(rng, options):
    return options[int(rng.integers(0, len(options)))]


def _eval_date(rng):
    return f"{int(rng.integers(2023, 2027)):04d}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}"


_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _generate_invoice_case(rng, index):
    vendor = generate_vendor_name(rng, descriptors=DESCRIPTOR_WORDS_EVAL, nouns=BUSINESS_NOUNS_EVAL,
                                   suffixes=VENDOR_SUFFIXES_EVAL, last_names=LAST_NAMES_EVAL)
    date = _eval_date(rng)
    subtotal = rng.uniform(50, 8000)
    tax_rate = rng.choice([5, 6, 7, 7.5, 8, 8.5, 9, 10])
    tax_amount = subtotal * tax_rate / 100
    total = subtotal + tax_amount

    header = _pick(rng, ["{vendor}", "Vendor: {vendor}", "From: {vendor}", "Billed By: {vendor}", "Supplier: {vendor}"])
    lines = [
        header.format(vendor=vendor),
        f"{_pick(rng, ['Invoice Date:', 'Date:', 'Issued:', 'Bill Date:'])} {date}",
        f"{_pick(rng, ['Tax ({r}%):', 'VAT ({r}%):', 'Sales Tax ({r}%):']).format(r=tax_rate)} ${tax_amount:,.2f}",
        f"{_pick(rng, ['Total Due:', 'Total:', 'Amount Due:', 'Grand Total:'])} ${total:,.2f}",
    ]
    fields = {
        "Supplier Name": vendor, "Transaction Date": date,
        "Total Amount Due": f"${total:,.2f}", "Tax Value (VAT/GST)": f"${tax_amount:,.2f}",
    }
    return f"gen_inv_{index:03d}.pdf", lines, "Invoice", fields


def _generate_receipt_case(rng, index):
    merchant = generate_merchant_name(rng, descriptors=MERCHANT_DESCRIPTORS_EVAL, nouns=MERCHANT_NOUNS_EVAL,
                                       last_names=LAST_NAMES_EVAL)
    date = _eval_date(rng)
    amount = rng.uniform(5, 300)
    method = _pick(rng, PAYMENT_METHODS)

    header = _pick(rng, ["{merchant}", "Merchant: {merchant}", "Sold By: {merchant}", "Store: {merchant}"])
    lines = [
        header.format(merchant=merchant),
        f"{_pick(rng, ['Purchase Date:', 'Date:', 'Transaction Date:'])} {date}",
        f"{_pick(rng, ['Amount Paid:', 'Total:', 'Amount:'])} ${amount:,.2f}",
        f"{_pick(rng, ['Payment Method:', 'Paid via:', 'Tender:'])} {method}",
    ]
    fields = {
        "Merchant": merchant, "Purchase Date": date,
        "Amount Paid": f"${amount:,.2f}", "Payment Method": method,
    }
    return f"gen_rec_{index:03d}.pdf", lines, "Receipt", fields


def _generate_statement_case(rng, index):
    holder = generate_account_holder_name(rng, first_names=FIRST_NAMES_EVAL, last_names=LAST_NAMES_EVAL)
    period = f"{_pick(rng, _MONTH_NAMES)} {int(rng.integers(2023, 2027))}"
    balance = rng.uniform(200, 15000)

    header = _pick(rng, ["{holder}", "Account Holder: {holder}", "Prepared For: {holder}", "Statement For: {holder}"])
    lines = [
        header.format(holder=holder),
        f"{_pick(rng, ['Statement Period:', 'Billing Period:', 'Period:'])} {period}",
        f"{_pick(rng, ['Closing Balance:', 'Ending Balance:', 'Balance:'])} ${balance:,.2f}",
    ]
    fields = {"Account Name": holder, "Statement Period": period, "Closing Balance": f"${balance:,.2f}"}
    return f"gen_stmt_{index:03d}.pdf", lines, "Statement", fields


def _generated_cases(count_per_type=17):
    rng = np.random.default_rng(_GEN_SEED)
    cases = []
    for i in range(count_per_type):
        cases.append(_generate_invoice_case(rng, i))
    for i in range(count_per_type):
        cases.append(_generate_receipt_case(rng, i))
    for i in range(count_per_type):
        cases.append(_generate_statement_case(rng, i))
    return cases


ALL_CASES = CASES + _generated_cases()


def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    ground_truth = {}
    for filename, lines, doc_type, fields in ALL_CASES:
        _make_pdf(filename, lines)
        ground_truth[filename] = {"type": doc_type, "fields": fields}

    with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Wrote {len(ALL_CASES)} evaluation documents to {DOCS_DIR}")
    print(f"Wrote ground truth to {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    main()
