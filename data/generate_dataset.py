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
    # Extra structural variety (length, punctuation, suffixes) so the model
    # learns to keep tagging a multi-word span through unfamiliar words
    # rather than memorizing this specific list of names.
    "Marsh & Fenwick Law Group", "Okoye & Partners Consulting",
    "Whitmore-Blackburn Industries", "Castellane-Vidal Freight",
    "Northbridge & Co", "Kestrel Ridge Holdings LLC", "Aurelia Textiles Inc",
    "Fairhaven Trust Company", "Bramblewood Timber & Sons",
    "Solenne Digital Media Group", "Thackeray Instruments Ltd",
    "Meridian & Lowell Architects", "Vantage Point Robotics",
    "Corvellis Manufacturing Co", "Hollowbrook Estate Vintners",
    "Panopticon Data Systems", "Ashgrove & Kane Insurance",
    "Larkspur Botanical Supply", "Windmere Shipping & Logistics",
    "Old Quarry Stoneworks", "Fennimore Publishing House",
    "Brightstone Capital Partners", "Ludlow & Ferris Machinery",
    "Wren Hollow Design Studio", "Castlemaine Foods International",
]

MERCHANTS = [
    "Corner Market", "Blue Bottle Coffee", "Riverside Diner", "Union Hardware",
    "Green Leaf Grocery", "Downtown Pharmacy", "Sunset Cinema", "Metro Fuel Stop",
    "Harbor Bookshop", "Peak Outdoor Gear", "Nightly Grill", "City Bike Rentals",
    "Old Town Bakery", "Lakeside Cafe", "Trailhead Sports",
    "Marlowe & Finch Booksellers", "The Gilded Spoon", "Northside Bike Co",
    "Pemberton's General Store", "Hearth & Kettle Cafe", "Saltmarsh Seafood Co",
    "Cobblestone Creamery", "Fox Hollow Nursery", "Ridgeline Outfitters",
    "The Tinker's Workshop", "Bellamy & Rowe Deli", "Windward Surf Shop",
    "Juniper Lane Florist", "Copper Kettle Roasters", "Thistledown Toy Shop",
]

ACCOUNT_HOLDERS = [
    "Jordan Whitfield", "Priya Anand", "Marcus Chen", "Elena Torres",
    "Samuel Okafor", "Grace Lindqvist", "David Nakamura", "Aisha Rahman",
    "Lucas Ferreira", "Nadia Kowalski", "Ethan Brooks", "Mei Lin",
    "Constance Beaumont-Reyes", "Julian Achterberg", "Rosalind Kwan-Mercer",
    "Theodore Vasilenko", "Bianca Odusanya-Croft", "Rafael Quintanilla",
    "Margarethe Solheim", "Anselm Okwuosa", "Delphine Marchetti-Hale",
    "Kwame Adebayo-Lindstrom", "Serafina Castellano", "Oisin Fitzgerald-Muto",
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


def _pick(rng, options):
    return options[int(rng.integers(0, len(options)))]


def _maybe(rng, p, line):
    """Include a boilerplate line only some of the time, so the model can't
    lean on one fixed document shape (a real invoice from a different vendor
    won't always carry the same surrounding lines)."""
    return [line] if rng.random() < p else []


INVOICE_HEADER_PHRASINGS = ["{vendor}", "Vendor: {vendor}", "From: {vendor}", "Billed By: {vendor}"]
INVOICE_DATE_PHRASINGS = ["Invoice Date:", "Date:", "Issued:"]
INVOICE_TOTAL_PHRASINGS = ["Total Due:", "Total:", "Amount Due:", "Grand Total:"]
INVOICE_TAX_PHRASINGS = ["Tax ({rate}%):", "VAT ({rate}%):", "Sales Tax ({rate}%):"]
INVOICE_CLOSINGS = [
    "Payment is due within 30 days. Thank you for your business.",
    "Thank you for your business.",
    "Please remit payment within 30 days.",
    "Questions about this invoice? Contact billing.",
]


def generate_invoice_document(rng):
    vendor = VENDORS[rng.integers(0, len(VENDORS))]
    date = _iso_date(rng) if rng.random() < 0.5 else _us_date(rng)
    subtotal = rng.uniform(100, 5000)
    tax_rate = rng.choice([5, 6, 7, 7.5, 8, 8.5, 9, 10])
    tax_amount = subtotal * (tax_rate / 100)
    total = subtotal + tax_amount

    header = _pick(rng, INVOICE_HEADER_PHRASINGS).format(vendor="{vendor}")
    header_text, header_vendor_text = header.split("{vendor}")[0], vendor

    lines = [
        _line((header_text, None), (header_vendor_text, "Vendor")) if header_text
        else _line((header_vendor_text, "Vendor")),
        *_maybe(rng, 0.7, _line((f"Invoice Number: {_invoice_number(rng)}", None))),
        _line((_pick(rng, INVOICE_DATE_PHRASINGS), None), (date, "Date")),
        *_maybe(rng, 0.6, _line(("Bill To: Customer Account #", None),
                                 (str(int(rng.integers(1000, 9999))), None))),
        *_maybe(rng, 0.8, _line((f"Subtotal: ${subtotal:,.2f}", None))),
        _line((_pick(rng, INVOICE_TAX_PHRASINGS).format(rate=tax_rate), None), (f"${tax_amount:,.2f}", "Tax")),
        _line((_pick(rng, INVOICE_TOTAL_PHRASINGS), None), (f"${total:,.2f}", "Total")),
        *_maybe(rng, 0.75, _line((_pick(rng, INVOICE_CLOSINGS), None))),
    ]
    return _assemble(lines)


RECEIPT_HEADER_PHRASINGS = ["{merchant}", "Merchant: {merchant}", "Sold By: {merchant}"]
RECEIPT_DATE_PHRASINGS = ["Purchase Date:", "Date:", "Transaction Date:"]
RECEIPT_TOTAL_PHRASINGS = ["Amount Paid:", "Total:", "Amount:"]
RECEIPT_METHOD_PHRASINGS = ["Payment Method:", "Paid via:", "Tender:"]
RECEIPT_CLOSINGS = [
    "Thank you for shopping with us.", "Come back soon!", "Have a great day.",
]


def generate_receipt_document(rng):
    merchant = MERCHANTS[int(rng.integers(0, len(MERCHANTS)))]
    date = _iso_date(rng) if rng.random() < 0.5 else _us_date(rng)
    amount = rng.uniform(5, 300)
    method = PAYMENT_METHODS[int(rng.integers(0, len(PAYMENT_METHODS)))]

    header = _pick(rng, RECEIPT_HEADER_PHRASINGS)
    header_text = header.split("{merchant}")[0]

    lines = [
        _line((header_text, None), (merchant, "Merchant")) if header_text
        else _line((merchant, "Merchant")),
        _line((_pick(rng, RECEIPT_DATE_PHRASINGS), None), (date, "Date")),
        *_maybe(rng, 0.6, _line(("Item Sundry goods x", None), (str(int(rng.integers(1, 6))), None))),
        _line((_pick(rng, RECEIPT_TOTAL_PHRASINGS), None), (f"${amount:,.2f}", "Total")),
        _line((_pick(rng, RECEIPT_METHOD_PHRASINGS), None), (method, "PaymentMethod")),
        *_maybe(rng, 0.7, _line((_pick(rng, RECEIPT_CLOSINGS), None))),
    ]
    return _assemble(lines)


STATEMENT_HEADER_PHRASINGS = ["{holder}", "Account Holder: {holder}", "Prepared For: {holder}"]
STATEMENT_PERIOD_PHRASINGS = ["Statement Period:", "Billing Period:", "Period:"]
STATEMENT_BALANCE_PHRASINGS = ["Closing Balance:", "Ending Balance:", "Balance:"]
STATEMENT_CLOSINGS = [
    "This statement is generated automatically each cycle.",
    "Please review your transactions and report any discrepancies.",
    "Thank you for banking with us.",
]


def generate_statement_document(rng):
    holder = ACCOUNT_HOLDERS[int(rng.integers(0, len(ACCOUNT_HOLDERS)))]
    period = _month_year(rng)
    balance = rng.uniform(200, 15000)

    header = _pick(rng, STATEMENT_HEADER_PHRASINGS)
    header_text = header.split("{holder}")[0]

    lines = [
        _line((header_text, None), (holder, "Account")) if header_text
        else _line((holder, "Account")),
        *_maybe(rng, 0.6, _line(("Account Summary", None))),
        _line((_pick(rng, STATEMENT_PERIOD_PHRASINGS), None), (period, "Period")),
        *_maybe(rng, 0.7, _line(("Opening Balance:", None), (f"${rng.uniform(100, 12000):,.2f}", None))),
        _line((_pick(rng, STATEMENT_BALANCE_PHRASINGS), None), (f"${balance:,.2f}", "Balance")),
        *_maybe(rng, 0.7, _line((_pick(rng, STATEMENT_CLOSINGS), None))),
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

    docs_per_type = 350
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
