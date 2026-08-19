"""Synthesize labeled invoice, receipt, and statement text for training,
since no real document corpus is available. Emits data/dataset.csv in a
CoNLL-style layout: one row per token, grouped by doc_id, columns:
doc_id,token,label.

This script has no dependency on the engine's model, only on its
tokenizer, so the labels line up exactly with how train.py will re-tokenize
raw text at inference time.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tokenizer import tokenize

RNG_SEED = 7

VENDORS = [
    "Acme Supply Co", "Blue Ocean Traders", "Nordic Timber Group",
    "Silverline Logistics", "Golden Gate Consulting", "Redwood Hardware",
    "Pioneer Freight Services", "Cobalt Software Solutions",
    "Harbor View Office Supplies", "Titan Steel Works",
    "Willow Creek Farms", "Sunrise Printing Co", "Delta Analytics Group",
    "Evergreen Landscaping", "Maple Leaf Bakery", "Quantum Electronics",
    "Ironclad Security Systems", "Bright Path Tutoring", "Cedarwood Furniture",
    "Alpine Water Co",
]

MERCHANTS = [
    "Corner Market", "Blue Bottle Coffee", "Riverside Diner", "Union Hardware",
    "Green Leaf Grocery", "Downtown Pharmacy", "Sunset Cinema", "Metro Fuel Stop",
    "Harbor Bookshop", "Peak Outdoor Gear", "Nightly Grill", "City Bike Rentals",
    "Old Town Bakery", "Lakeside Cafe", "Trailhead Sports",
]

ACCOUNT_HOLDERS = [
    "Jordan Whitfield", "Priya Anand", "Marcus Chen", "Elena Torres",
    "Samuel Okafor", "Grace Lindqvist", "David Nakamura", "Aisha Rahman",
    "Lucas Ferreira", "Nadia Kowalski", "Ethan Brooks", "Mei Lin",
]

PAYMENT_METHODS = ["Cash", "Visa", "Mastercard", "Amex", "PayPal", "Debit"]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTHS_31 = [1, 3, 5, 7, 8, 10, 12]


def _iso_date(rng):
    year = rng.integers(2023, 2027)
    month = int(rng.integers(1, 13))
    day_max = 31 if month in MONTHS_31 else (28 if month == 2 else 30)
    day = int(rng.integers(1, day_max + 1))
    return f"{year:04d}-{month:02d}-{day:02d}"


def _us_date(rng):
    year = rng.integers(2023, 2027)
    month = int(rng.integers(1, 13))
    day_max = 31 if month in MONTHS_31 else (28 if month == 2 else 30)
    day = int(rng.integers(1, day_max + 1))
    return f"{month:02d}/{day:02d}/{year:04d}"


def _month_year(rng):
    month = MONTH_NAMES[int(rng.integers(0, 12))]
    year = int(rng.integers(2023, 2027))
    return f"{month} {year}"


def _money(rng, lo, hi):
    val = rng.uniform(lo, hi)
    return f"${val:,.2f}"


def _invoice_number(rng):
    return f"INV-{int(rng.integers(10000, 99999))}"


# Entities whose values can span more than one token (the tokenizer never
# collapses a name into a single token, unlike dates/money/percentages).
MULTI_TOKEN_ENTITIES = {"Vendor", "Merchant", "Account", "PaymentMethod", "Period"}


def _labeled(text, entity):
    """Tokenize `text` and tag it as a run of `entity` (or plain 'O')."""
    tokens = tokenize(text)
    if not tokens:
        return [], []
    if entity is None:
        labels = ["O"] * len(tokens)
    elif entity in MULTI_TOKEN_ENTITIES:
        labels = [f"B-{entity}"] + [f"I-{entity}"] * (len(tokens) - 1)
    else:
        # Date / Total / Tax / Balance are single-token spans by construction
        # (the tokenizer's date/money regexes swallow the whole value).
        labels = [f"B-{entity}"] + ["O"] * (len(tokens) - 1)
    return tokens, labels


def _line(*pieces):
    """pieces: list of (text, entity_or_None). Returns (tokens, labels)."""
    all_tokens, all_labels = [], []
    for text, entity in pieces:
        toks, labs = _labeled(text, entity)
        all_tokens.extend(toks)
        all_labels.extend(labs)
    return all_tokens, all_labels


def _assemble(lines):
    doc_tokens, doc_labels = [], []
    for toks, labs in lines:
        doc_tokens.extend(toks)
        doc_labels.extend(labs)
    return doc_tokens, doc_labels


def generate_invoice_document(rng):
    vendor = VENDORS[rng.integers(0, len(VENDORS))]
    date = _iso_date(rng) if rng.random() < 0.5 else _us_date(rng)
    subtotal = rng.uniform(100, 5000)
    tax_rate = rng.choice([5, 6, 7, 7.5, 8, 8.5, 9, 10])
    tax_amount = subtotal * (tax_rate / 100)
    total = subtotal + tax_amount

    lines = [
        _line((vendor, "Vendor")),
        _line((f"Invoice Number: {_invoice_number(rng)}", None)),
        _line(("Invoice Date:", None), (date, "Date")),
        _line(("Bill To: Customer Account #", None), (str(int(rng.integers(1000, 9999))), None)),
        _line((f"Subtotal: ${subtotal:,.2f}", None)),
        _line((f"Tax ({tax_rate}%):", None), (f"${tax_amount:,.2f}", "Tax")),
        _line(("Total Due:", None), (f"${total:,.2f}", "Total")),
        _line(("Payment is due within 30 days. Thank you for your business.", None)),
    ]
    return _assemble(lines)


def generate_receipt_document(rng):
    merchant = MERCHANTS[int(rng.integers(0, len(MERCHANTS)))]
    date = _iso_date(rng) if rng.random() < 0.5 else _us_date(rng)
    amount = rng.uniform(5, 300)
    method = PAYMENT_METHODS[int(rng.integers(0, len(PAYMENT_METHODS)))]

    lines = [
        _line((merchant, "Merchant")),
        _line(("Purchase Date:", None), (date, "Date")),
        _line(("Item Sundry goods x", None), (str(int(rng.integers(1, 6))), None)),
        _line(("Amount Paid:", None), (f"${amount:,.2f}", "Total")),
        _line(("Payment Method:", None), (method, "PaymentMethod")),
        _line(("Thank you for shopping with us.", None)),
    ]
    return _assemble(lines)


def generate_statement_document(rng):
    holder = ACCOUNT_HOLDERS[int(rng.integers(0, len(ACCOUNT_HOLDERS)))]
    period = _month_year(rng)
    balance = rng.uniform(200, 15000)

    lines = [
        _line((holder, "Account")),
        _line(("Account Summary", None)),
        _line(("Statement Period:", None), (period, "Period")),
        _line(("Opening Balance:", None), (f"${rng.uniform(100, 12000):,.2f}", None)),
        _line(("Closing Balance:", None), (f"${balance:,.2f}", "Balance")),
        _line(("This statement is generated automatically each cycle.", None)),
    ]
    return _assemble(lines)


DOCUMENT_GENERATORS = [
    generate_invoice_document,
    generate_receipt_document,
    generate_statement_document,
]


def main():
    import numpy as np

    rng = np.random.default_rng(RNG_SEED)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

    docs_per_type = 220
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "token", "label"])
        doc_id = 0
        for generator in DOCUMENT_GENERATORS:
            for _ in range(docs_per_type):
                tokens, labels = generator(rng)
                for tok, lab in zip(tokens, labels):
                    writer.writerow([doc_id, tok, lab])
                doc_id += 1

    total = docs_per_type * len(DOCUMENT_GENERATORS)
    print(f"Wrote {total} synthetic documents ({docs_per_type} each of "
          f"invoice/receipt/statement) to {out_path}")


if __name__ == "__main__":
    main()
