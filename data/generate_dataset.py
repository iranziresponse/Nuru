"""Synthesize labeled invoice, receipt, and statement text for training,
since no real document corpus is available. Emits data/dataset.csv in a
CoNLL-style layout: one row per token, grouped by doc_id, columns:
doc_id,token,label.

This script has no dependency on the engine's model, only on its
tokenizer, so the labels line up exactly with how train.py will re-tokenize
raw text at inference time.

Names are built combinatorially from small word-fragment pools rather than
hand-typed as literal lists: a fixed, seeded 85/15 split of every pool at
module load produces a *_TRAIN half (drawn from here) and a *_EVAL half
(drawn from eval/generate_eval_set.py), so "deliberately unseen from
training" is a structural guarantee of the split itself, not something a
human has to remember while adding new names by hand.

Real invoices are full of content that isn't any of the fields we tag:
addresses, phone/email lines, item descriptions, terms-and-conditions
boilerplate, letterhead taglines. That content is generated here too and
spliced in at a randomized position (not always the same trailing slot),
so the model learns to find entities embedded in surrounding text it
hasn't memorized the shape of, rather than always in the same place.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tokenizer import tokenize

RNG_SEED = 7
_SPLIT_SEED = 20260101  # fixed, independent of RNG_SEED, so the *_TRAIN/*_EVAL split never drifts with generation changes


def _split_pool(pool, eval_fraction=0.15):
    """Deterministic 85/15 shuffle-then-slice, independent of whatever RNG
    the caller is using to generate documents, so the split itself never
    changes even if generation logic or its RNG usage does."""
    import numpy as np

    rng = np.random.default_rng(_SPLIT_SEED + len(pool))  # vary by pool size so different pools don't shuffle identically
    indices = np.arange(len(pool))
    rng.shuffle(indices)
    n_eval = max(1, int(len(pool) * eval_fraction))
    eval_idx = set(indices[:n_eval].tolist())
    train = [pool[i] for i in range(len(pool)) if i not in eval_idx]
    held_out = [pool[i] for i in range(len(pool)) if i in eval_idx]
    return train, held_out


# ---- word-fragment pools ------------------------------------------------

_DESCRIPTOR_WORDS = [
    "Bright", "Silver", "North", "Harbor", "Cedar", "Kestrel", "Meridian", "Iron",
    "Copper", "Amber", "Falcon", "Granite", "Ember", "Frost", "Willow", "Sable",
    "Onyx", "Cobalt", "Vermillion", "Lantern", "Highland", "Coastal", "Union",
    "Crestwood", "Ashgrove", "Thornwood", "Blackstone", "Whitfield", "Larkspur",
    "Marlowe", "Sterling", "Cambridge", "Ridgemont", "Fairview", "Brookhaven",
    "Ironwood", "Stonebridge", "Windward", "Elmsworth", "Hawthorne", "Summit",
    "Anchor", "Beacon", "Cascade", "Driftwood", "Evergreen", "Foxglove", "Gable",
    "Hollow", "Ivory", "Juniper", "Kingsley", "Lockwood", "Millbrook",
    "Nightingale", "Overlook", "Pinecrest", "Quarrystone", "Riverbend", "Sundale",
]

_BUSINESS_NOUNS = [
    "Supply", "Trading", "Freight", "Consulting", "Systems", "Instruments",
    "Vintners", "Logistics", "Holdings", "Industries", "Manufacturing",
    "Robotics", "Analytics", "Textiles", "Foods", "Publishing", "Shipping",
    "Machinery", "Design Studio", "Data Systems", "Botanical Supply",
    "Stoneworks", "Capital Partners", "Insurance", "Architects", "Media Group",
    "Software Solutions", "Hardware", "Electronics", "Furniture", "Landscaping",
    "Bakery", "Water Co", "Security Systems", "Tutoring", "Farms", "Printing Co",
    "Office Supplies", "Steel Works", "Timber Group", "Legal Services",
    "Fabrication Co", "Vineyards", "Financial Group", "Engineering", "Ventures",
    "Technologies", "Solutions", "Materials", "Distribution", "Packaging",
    "Chemicals", "Energy", "Construction", "Realty", "Transport", "Foodservice",
    "Apparel", "Pharmaceuticals", "Automotive", "Aerospace", "Telecom",
    "Networks", "Labs", "Imports", "Exports",
]

_VENDOR_SUFFIXES = [
    "Co", "Inc", "LLC", "Group", "Partners", "& Sons", "Holdings", "Industries",
    "Ltd", "Corp", "International", "Enterprises", "Associates", "Trust Company",
]

_LAST_NAMES = [
    "Whitfield", "Anand", "Chen", "Torres", "Okafor", "Lindqvist", "Nakamura",
    "Rahman", "Ferreira", "Kowalski", "Brooks", "Lin", "Beaumont", "Achterberg",
    "Kwan", "Vasilenko", "Odusanya", "Quintanilla", "Solheim", "Okwuosa",
    "Marchetti", "Adebayo", "Castellano", "Fitzgerald", "Marsh", "Fenwick",
    "Okoye", "Whitmore", "Blackburn", "Castellane", "Vidal", "Northbridge",
    "Kestrel", "Aurelia", "Fairhaven", "Bramblewood", "Solenne", "Thackeray",
    "Meridian", "Lowell", "Corvellis", "Hollowbrook", "Panopticon", "Ashgrove",
    "Kane", "Windmere", "Ludlow", "Ferris", "Hollow", "Castlemaine", "Hargrove",
    "Sinclair", "Vance", "Delacroix", "Osei", "Yamamoto", "Petrov", "Almeida",
    "Bergstrom", "Cavanagh", "Doyle", "Eriksson", "Falk", "Guzman", "Halvorsen",
    "Ibsen", "Jarvinen", "Kessler", "Larkin", "Moreau", "Naidoo", "Ostrowski",
    "Pemberton", "Quintero", "Redgrave", "Sundqvist", "Tremblay", "Ueda",
    "Voss", "Winterbourne", "Yilmaz", "Zhukov",
]

_FIRST_NAMES = [
    "Jordan", "Priya", "Marcus", "Elena", "Samuel", "Grace", "David", "Aisha",
    "Lucas", "Nadia", "Ethan", "Mei", "Constance", "Julian", "Rosalind",
    "Theodore", "Bianca", "Rafael", "Margarethe", "Anselm", "Delphine", "Kwame",
    "Serafina", "Oisin", "Amara", "Dimitri", "Freya", "Hassan", "Ingrid",
    "Javier", "Katarina", "Liam", "Mireille", "Noor", "Oscar", "Petra",
    "Quentin", "Ravi", "Sofia", "Tobias", "Ursula", "Viktor", "Wren", "Ximena",
    "Yusuf", "Zara", "Adrian", "Beatrix", "Cyrus", "Daniela", "Emeka", "Fiona",
    "Gideon", "Helena", "Idris", "Jasmine", "Kenji", "Leilani", "Magnus",
    "Naomi", "Otis", "Paloma", "Rune", "Selin", "Tamsin", "Umberto",
]

_MERCHANT_DESCRIPTORS = [
    "Corner", "Riverside", "Downtown", "Sunset", "Metro", "Harbor", "Peak",
    "Old Town", "Lakeside", "Northside", "Cobblestone", "Fox Hollow", "The Gilded",
    "Green Leaf", "Union", "Nightly", "City", "Trailhead", "Saltmarsh",
    "Windward", "Juniper Lane", "Copper Kettle", "Thistledown", "Maple Street",
    "Willow Creek", "Cedar Point", "Redbrick", "Ivy Lane", "Sparrow",
    "Amber Coast", "Driftwood", "Hollybrook", "Elmwood", "Rosewood", "Birchwood",
]

_MERCHANT_NOUNS = [
    "Market", "Coffee", "Diner", "Hardware", "Grocery", "Pharmacy", "Cinema",
    "Fuel Stop", "Bookshop", "Outdoor Gear", "Grill", "Bike Rentals", "Bakery",
    "Cafe", "Sports", "Booksellers", "General Store", "Deli", "Surf Shop",
    "Florist", "Roasters", "Toy Shop", "Creamery", "Nursery", "Outfitters",
    "Workshop", "Boutique", "Tavern", "Bistro", "Barbershop", "Salon", "Studio",
    "Gallery", "Antiques", "Garden Center", "Butcher", "Wine Shop", "Tea House",
]

_STREET_NAMES = [
    "Maple", "Elm", "Cedar", "Birch", "Willow", "Oak", "Harbor", "Ridge",
    "Sunset", "Meadow", "Highland", "River", "Orchard", "Chestnut", "Pine",
    "Aspen", "Juniper", "Magnolia", "Sycamore", "Hillcrest", "Fairview",
    "Lakeview", "Brookside", "Foxglove", "Stonebridge",
]
_STREET_SUFFIXES = ["Street", "Avenue", "Lane", "Drive", "Boulevard", "Way", "Road", "Court"]
_CITIES = [
    ("Springfield", "IL"), ("Rivertown", "OR"), ("Fairhaven", "MA"), ("Lakewood", "CO"),
    ("Brookfield", "WI"), ("Cedar Falls", "IA"), ("Millbrook", "NY"), ("Ashland", "OH"),
    ("Elmwood", "TX"), ("Riverside", "CA"), ("Highland Park", "NJ"), ("Stonegate", "GA"),
]

_ITEM_NAMES = [
    "Consulting Services (per hour)", "Widget Assembly Kit", "Premium Support Plan",
    "Replacement Parts Set", "On-Site Installation", "Custom Design Package",
    "Bulk Paper Supply", "Office Chair", "Networking Cable (10m)", "Software License",
    "Maintenance Visit", "Delivery Fee", "Rush Processing Fee", "Training Session",
    "Equipment Rental (weekly)", "Cleaning Service", "Freight Handling",
]

_TERMS_BOILERPLATE = [
    "Payment is due within 15 days of the invoice date.",
    "A late fee of 1.5% per month applies to overdue balances.",
    "Please include the invoice number with your payment.",
    "All sales are final unless otherwise noted.",
    "Prices are subject to change without notice.",
    "This document is a computer-generated record and requires no signature.",
    "Please retain this document for your records.",
    "Disputes must be raised within 30 days of receipt.",
]

_TAGLINES = [
    "An ISO 9001 Certified Company", "Serving clients since 1998",
    "Locally owned and operated", "Quality. Integrity. Excellence.",
    "Proud member of the Chamber of Commerce", "Family owned since 2004",
    "Committed to sustainable practices", "Your satisfaction is our priority",
]

# Fixed, seeded split so training and eval draw disjoint identifying name
# fragments (the part of a name that makes it *this* company/person, not a
# generic one). Only DESCRIPTOR_WORDS/LAST_NAMES/FIRST_NAMES are split:
# these are effectively an open-ended set of novel proper-noun-like words,
# so holding some out genuinely tests generalization to an unseen name.
#
# BUSINESS_NOUNS/MERCHANT_NOUNS/VENDOR_SUFFIXES are deliberately NOT split
# and stay fully shared between training and eval. These are a small,
# closed vocabulary of ordinary business-type words ("Butcher", "Co",
# "Workshop"): holding 15% of them out of training entirely would mean the
# model never saw those specific words used as an I-Vendor/I-Merchant
# continuation in any context at all, which isn't a fair generalization
# test, it's testing on a word the model had no way to learn regardless of
# capability. (This was tried and measured: it made eval field accuracy on
# multi-word names *worse* than before this whole change, concentrated
# almost entirely on spans ending in a noun/suffix that had been held out
# of training this way. Every held-out generalization test here should
# come from combining familiar structural vocabulary with a genuinely new
# identifying word, not from vocabulary the model was never shown.)
DESCRIPTOR_WORDS_TRAIN, DESCRIPTOR_WORDS_EVAL = _split_pool(_DESCRIPTOR_WORDS)
LAST_NAMES_TRAIN, LAST_NAMES_EVAL = _split_pool(_LAST_NAMES)
FIRST_NAMES_TRAIN, FIRST_NAMES_EVAL = _split_pool(_FIRST_NAMES)
MERCHANT_DESCRIPTORS_TRAIN, MERCHANT_DESCRIPTORS_EVAL = _split_pool(_MERCHANT_DESCRIPTORS)
STREET_NAMES_TRAIN, STREET_NAMES_EVAL = _split_pool(_STREET_NAMES)

BUSINESS_NOUNS_TRAIN = BUSINESS_NOUNS_EVAL = _BUSINESS_NOUNS
VENDOR_SUFFIXES_TRAIN = VENDOR_SUFFIXES_EVAL = _VENDOR_SUFFIXES
MERCHANT_NOUNS_TRAIN = MERCHANT_NOUNS_EVAL = _MERCHANT_NOUNS

PAYMENT_METHODS = ["Cash", "Visa", "Mastercard", "Amex", "PayPal", "Debit"]
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTHS_31 = [1, 3, 5, 7, 8, 10, 12]


def generate_vendor_name(rng, descriptors=DESCRIPTOR_WORDS_TRAIN, nouns=BUSINESS_NOUNS_TRAIN,
                          suffixes=VENDOR_SUFFIXES_TRAIN, last_names=LAST_NAMES_TRAIN):
    """Combines word-fragment pools into a vendor-style business name. About
    a third of the time builds a "Surname & Surname Noun" firm-style name
    instead, since that ampersand pattern (a real, common business-name
    shape) is exactly what previously caused multi-word span truncation:
    the model needs many examples of tagging straight through an "&" to
    learn to keep going instead of stopping at the first surname."""
    style = rng.random()
    if style < 0.32:
        a, b = last_names[int(rng.integers(0, len(last_names)))], last_names[int(rng.integers(0, len(last_names)))]
        while b == a:
            b = last_names[int(rng.integers(0, len(last_names)))]
        noun = nouns[int(rng.integers(0, len(nouns)))]
        return f"{a} & {b} {noun}"
    descriptor = descriptors[int(rng.integers(0, len(descriptors)))]
    noun = nouns[int(rng.integers(0, len(nouns)))]
    name = f"{descriptor} {noun}"
    if rng.random() < 0.45:
        name += f" {suffixes[int(rng.integers(0, len(suffixes)))]}"
    return name


def generate_merchant_name(rng, descriptors=MERCHANT_DESCRIPTORS_TRAIN, nouns=MERCHANT_NOUNS_TRAIN,
                            last_names=LAST_NAMES_TRAIN):
    if rng.random() < 0.2:
        a, b = last_names[int(rng.integers(0, len(last_names)))], last_names[int(rng.integers(0, len(last_names)))]
        while b == a:
            b = last_names[int(rng.integers(0, len(last_names)))]
        noun = nouns[int(rng.integers(0, len(nouns)))]
        return f"{a} & {b} {noun}"
    descriptor = descriptors[int(rng.integers(0, len(descriptors)))]
    noun = nouns[int(rng.integers(0, len(nouns)))]
    return f"{descriptor} {noun}"


def generate_account_holder_name(rng, first_names=FIRST_NAMES_TRAIN, last_names=LAST_NAMES_TRAIN):
    first = first_names[int(rng.integers(0, len(first_names)))]
    if rng.random() < 0.2:
        a, b = last_names[int(rng.integers(0, len(last_names)))], last_names[int(rng.integers(0, len(last_names)))]
        while b == a:
            b = last_names[int(rng.integers(0, len(last_names)))]
        return f"{first} {a}-{b}"
    last = last_names[int(rng.integers(0, len(last_names)))]
    return f"{first} {last}"


def generate_address(rng, street_names=STREET_NAMES_TRAIN):
    number = int(rng.integers(100, 9999))
    street = street_names[int(rng.integers(0, len(street_names)))]
    suffix = _STREET_SUFFIXES[int(rng.integers(0, len(_STREET_SUFFIXES)))]
    city, state = _CITIES[int(rng.integers(0, len(_CITIES)))]
    zip_code = int(rng.integers(10000, 99999))
    return f"{number} {street} {suffix}, {city}, {state} {zip_code}"


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


def _phone(rng):
    return f"+1 ({int(rng.integers(200, 999))}) {int(rng.integers(200, 999))}-{int(rng.integers(1000, 9999))}"


def _email(rng, name):
    handle = "".join(ch for ch in name.lower() if ch.isalnum())[:16] or "contact"
    domain = _pick(rng, ["company.com", "example.com", "business.net", "mail.co"])
    return f"info@{handle}.{domain.split('.')[-1]}"


def _website(rng, name):
    handle = "".join(ch for ch in name.lower() if ch.isalnum())[:16] or "company"
    return f"www.{handle}.com"


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


def _noise_lines(rng, name_for_contact):
    """0-4 lines of realistic non-entity content real documents carry
    (address, phone/email/website, an item line, terms boilerplate, a
    tagline), each independently. Every piece is plain 'O' label, since
    none of it is a field this project extracts."""
    candidates = [
        (0.5, _line((generate_address(rng), None))),
        (0.35, _line((_phone(rng), None))),
        (0.3, _line((_email(rng, name_for_contact), None))),
        (0.25, _line((_website(rng, name_for_contact), None))),
        (0.3, _line((f"{int(rng.integers(1, 6))} x {_pick(rng, _ITEM_NAMES)}", None))),
        (0.35, _line((_pick(rng, _TERMS_BOILERPLATE), None))),
        (0.25, _line((_pick(rng, _TAGLINES), None))),
    ]
    lines = []
    for p, line in candidates:
        lines.extend(_maybe(rng, p, line))
    return lines[:4]


def _interleave_noise(rng, core_lines, noise_lines):
    """Splices noise lines into the document at randomized positions
    (before, between, or after any core line), one at a time, rather than
    always appending them at the end: a real invoice's letterhead can sit
    above the vendor name, its terms can sit before or after the total,
    and the model needs to see entities embedded in text on both sides,
    not only ever preceded and never followed by boilerplate."""
    result = list(core_lines)
    for noise_line in noise_lines:
        pos = int(rng.integers(0, len(result) + 1))
        result.insert(pos, noise_line)
    return result


INVOICE_HEADER_PHRASINGS = [
    "{vendor}", "Vendor: {vendor}", "From: {vendor}", "Billed By: {vendor}",
    "Supplier: {vendor}", "Issued By: {vendor}", "Sold By: {vendor}",
    "Remit To: {vendor}",
]
INVOICE_DATE_PHRASINGS = [
    "Invoice Date:", "Date:", "Issued:", "Bill Date:", "Date Issued:", "Order Date:",
]
INVOICE_TOTAL_PHRASINGS = [
    "Total Due:", "Total:", "Amount Due:", "Grand Total:", "Balance Due:",
    "Net Total:", "Invoice Total:", "Total Payable:",
]
INVOICE_TAX_PHRASINGS = [
    "Tax ({rate}%):", "VAT ({rate}%):", "Sales Tax ({rate}%):", "GST ({rate}%):",
    "Tax Rate ({rate}%):",
]
INVOICE_CLOSINGS = [
    "Payment is due within 30 days. Thank you for your business.",
    "Thank you for your business.",
    "Please remit payment within 30 days.",
    "Questions about this invoice? Contact billing.",
    "We appreciate your prompt payment.",
    "This invoice was generated automatically.",
]


def generate_invoice_document(rng, vendor_pool=None):
    vendor = generate_vendor_name(rng) if vendor_pool is None else _pick(rng, vendor_pool)
    date = _iso_date(rng) if rng.random() < 0.5 else _us_date(rng)
    subtotal = rng.uniform(100, 5000)
    tax_rate = rng.choice([5, 6, 7, 7.5, 8, 8.5, 9, 10])
    tax_amount = subtotal * (tax_rate / 100)
    total = subtotal + tax_amount

    header = _pick(rng, INVOICE_HEADER_PHRASINGS).format(vendor="{vendor}")
    header_text, header_vendor_text = header.split("{vendor}")[0], vendor

    core = [
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
    lines = _interleave_noise(rng, core, _noise_lines(rng, vendor))
    return _assemble(lines)


RECEIPT_HEADER_PHRASINGS = [
    "{merchant}", "Merchant: {merchant}", "Sold By: {merchant}", "Store: {merchant}",
    "Purchased At: {merchant}",
]
RECEIPT_DATE_PHRASINGS = [
    "Purchase Date:", "Date:", "Transaction Date:", "Sale Date:", "Date of Purchase:",
]
RECEIPT_TOTAL_PHRASINGS = [
    "Amount Paid:", "Total:", "Amount:", "Total Paid:", "Charged:", "Total Charged:",
]
RECEIPT_METHOD_PHRASINGS = [
    "Payment Method:", "Paid via:", "Tender:", "Payment Type:", "Method:",
]
RECEIPT_CLOSINGS = [
    "Thank you for shopping with us.", "Come back soon!", "Have a great day.",
    "We appreciate your business.", "See you next time!",
]


def generate_receipt_document(rng, merchant_pool=None):
    merchant = generate_merchant_name(rng) if merchant_pool is None else _pick(rng, merchant_pool)
    date = _iso_date(rng) if rng.random() < 0.5 else _us_date(rng)
    amount = rng.uniform(5, 300)
    method = PAYMENT_METHODS[int(rng.integers(0, len(PAYMENT_METHODS)))]

    header = _pick(rng, RECEIPT_HEADER_PHRASINGS)
    header_text = header.split("{merchant}")[0]

    core = [
        _line((header_text, None), (merchant, "Merchant")) if header_text
        else _line((merchant, "Merchant")),
        _line((_pick(rng, RECEIPT_DATE_PHRASINGS), None), (date, "Date")),
        *_maybe(rng, 0.6, _line(("Item Sundry goods x", None), (str(int(rng.integers(1, 6))), None))),
        _line((_pick(rng, RECEIPT_TOTAL_PHRASINGS), None), (f"${amount:,.2f}", "Total")),
        _line((_pick(rng, RECEIPT_METHOD_PHRASINGS), None), (method, "PaymentMethod")),
        *_maybe(rng, 0.7, _line((_pick(rng, RECEIPT_CLOSINGS), None))),
    ]
    lines = _interleave_noise(rng, core, _noise_lines(rng, merchant))
    return _assemble(lines)


STATEMENT_HEADER_PHRASINGS = [
    "{holder}", "Account Holder: {holder}", "Prepared For: {holder}",
    "Statement For: {holder}", "Customer: {holder}",
]
STATEMENT_PERIOD_PHRASINGS = [
    "Statement Period:", "Billing Period:", "Period:", "Cycle:", "Reporting Period:",
]
STATEMENT_BALANCE_PHRASINGS = [
    "Closing Balance:", "Ending Balance:", "Balance:", "Current Balance:",
    "Balance Due:", "Statement Balance:",
]
STATEMENT_CLOSINGS = [
    "This statement is generated automatically each cycle.",
    "Please review your transactions and report any discrepancies.",
    "Thank you for banking with us.",
    "Contact customer service with any questions about this statement.",
]


def generate_statement_document(rng, holder_pool=None):
    holder = generate_account_holder_name(rng) if holder_pool is None else _pick(rng, holder_pool)
    period = _month_year(rng)
    balance = rng.uniform(200, 15000)

    header = _pick(rng, STATEMENT_HEADER_PHRASINGS)
    header_text = header.split("{holder}")[0]

    core = [
        _line((header_text, None), (holder, "Account")) if header_text
        else _line((holder, "Account")),
        *_maybe(rng, 0.6, _line(("Account Summary", None))),
        _line((_pick(rng, STATEMENT_PERIOD_PHRASINGS), None), (period, "Period")),
        *_maybe(rng, 0.7, _line(("Opening Balance:", None), (f"${rng.uniform(100, 12000):,.2f}", None))),
        _line((_pick(rng, STATEMENT_BALANCE_PHRASINGS), None), (f"${balance:,.2f}", "Balance")),
        *_maybe(rng, 0.7, _line((_pick(rng, STATEMENT_CLOSINGS), None))),
    ]
    lines = _interleave_noise(rng, core, _noise_lines(rng, holder))
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

    docs_per_type = 1500
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
