"""Weak/fuzzy alignment: given a real document's OCR'd tokens and a known
ground-truth field value (from a real, human-labeled dataset like SROIE),
find which span of tokens that value actually corresponds to, so it can
be turned into BIO training labels the same shape as the synthetic
generator produces.

This exists because real OCR text never contains the ground truth value
verbatim: "MR D.I.Y." comes back as "MR O.1.�." Exact string matching
would find nothing on the majority of real documents; fuzzy matching
finds the right span most of the time, at the cost of occasionally
finding the wrong one or nothing at all. Every function here is
deliberately conservative (a high similarity threshold, exact-enough
numeric matching for money/dates) and returns None rather than a weak
guess when it isn't confident, because a wrong label actively teaches
the model the wrong thing, while a missing label just means one fewer
training example for that field on that document, the same as when a
synthetic document's `_maybe()` omits a line.

No dependency on the trained model, only on engine.tokenizer, so this
stays reusable for any real dataset's ground truth, not just SROIE's.
"""

import difflib
import re

from engine.tokenizer import normalize

_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def _normalize_for_match(text):
    return _ALNUM_RE.sub("", text.lower())


def find_best_text_span(tokens, target_text, max_span_len=14, min_ratio=0.55, min_coverage=0.6):
    """Finds the contiguous token span whose joined text best fuzzy-matches
    target_text (case/punctuation-insensitive). Used for name-like fields
    (a company/merchant name), where OCR noise is heaviest and only
    approximate string similarity, not exact structure, can find it.

    max_span_len is measured in *tokens*, not words: punctuation is its
    own token ("BOOK TA .K (TAMAN DAYA) SDN BHD" is 10 tokens for 7
    words), so a low limit here silently caps the search well short of a
    real multi-word name and produces a truncated-but-plausible-looking
    match instead of the true, longer span - this was found and fixed by
    measuring, not assumed: an earlier max_span_len=6 was passing its own
    unit tests (which used short examples) while quietly truncating real
    SROIE merchant names during actual alignment. min_coverage additionally
    rejects any candidate shorter than 60% of the target's own length,
    so a short, spuriously-well-matching substring can't outscore the
    real (longer, still-legitimately-similar) span.

    Returns (start, end_exclusive, ratio) for the highest-scoring
    qualifying span, or None if nothing clears both bars. O(n *
    max_span_len) token comparisons, fine at real-receipt token counts.
    """
    target_norm = _normalize_for_match(target_text)
    if not target_norm:
        return None
    min_len = max(1, int(len(target_norm) * min_coverage))
    best = None
    n = len(tokens)
    for start in range(n):
        for length in range(1, max_span_len + 1):
            end = start + length
            if end > n:
                break
            candidate_norm = _normalize_for_match("".join(tokens[start:end]))
            if len(candidate_norm) < min_len:
                continue
            ratio = difflib.SequenceMatcher(None, candidate_norm, target_norm).ratio()
            if best is None or ratio > best[2]:
                best = (start, end, ratio)
    if best and best[2] >= min_ratio:
        return best
    return None


def _date_number_parts(text):
    return re.findall(r"\d+", text)


def _date_parts_match(candidate_parts, target_parts):
    """Compares two dates part-by-part (day/month/year in whatever order
    the original punctuation implied), allowing a 2-digit year to match
    a 4-digit year on the same underlying year (19 vs 2019), since
    SROIE's own ground truth mixes both within the same field across
    documents. Day/month values are always < 32 so this rule never
    accidentally lets an unrelated day or month pair through."""
    if len(candidate_parts) != len(target_parts):
        return False
    for a, b in zip(candidate_parts, target_parts):
        ia, ib = int(a), int(b)
        if ia == ib:
            continue
        if (len(a) == 4 or len(b) == 4) and ia % 100 == ib % 100:
            continue
        return False
    return True


def find_date_span(tokens, target_date):
    """Finds a single <DATE>-shaped token whose day/month/year parts match
    target_date's. Deliberately does not fuzzy-match the surrounding
    punctuation or format (SROIE alone uses "25/12/2018", "19-03-2018",
    and "24 MAR 18" for the same underlying concept) - only the numeric
    parts have to agree, and only a token the tokenizer already
    recognizes as date-shaped is considered, so this never mislabels an
    unrelated number as a date.

    Returns (start, end_exclusive) for the first matching token, or None.
    """
    target_parts = _date_number_parts(target_date)
    if len(target_parts) != 3:
        return None
    for i, tok in enumerate(tokens):
        if normalize(tok) != "<DATE>":
            continue
        candidate_parts = _date_number_parts(tok)
        if len(candidate_parts) == 3 and _date_parts_match(candidate_parts, target_parts):
            return (i, i + 1)
    return None


def _parse_money_loosely(text):
    cleaned = re.sub(r"[^0-9.]", "", text)
    if not cleaned or cleaned.count(".") > 1:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def find_money_span(tokens, target_amount, tolerance=0.01):
    """Finds a single <MONEY>-shaped token whose numeric value matches
    target_amount (a string like "9.00") within tolerance. Like dates,
    only a token the tokenizer already recognizes as money-shaped is
    considered, and the match has to be numerically exact (up to
    floating-point tolerance), not fuzzy: a wrong total is a wrong
    training label, and money is the one field real OCR tends to get
    right when it recognizes a number at all.

    Returns (start, end_exclusive), or None.
    """
    target_value = _parse_money_loosely(target_amount)
    if target_value is None:
        return None
    for i, tok in enumerate(tokens):
        if normalize(tok) != "<MONEY>":
            continue
        candidate_value = _parse_money_loosely(tok)
        if candidate_value is not None and abs(candidate_value - target_value) < tolerance:
            return (i, i + 1)
    return None


def build_bio_labels(tokens, spans_by_entity):
    """spans_by_entity: {"Merchant": (start, end_exclusive), ...}. Returns
    a labels list the same length as tokens, "O" everywhere except each
    given span, tagged B-<entity> then I-<entity> for the rest of that
    span. Spans must not overlap (the caller is responsible for that;
    this project's three real-data fields - Merchant/Date/Total - never
    overlap in practice since they're matched by disjoint criteria)."""
    labels = ["O"] * len(tokens)
    for entity, span in spans_by_entity.items():
        if span is None:
            continue
        start, end = span
        for i in range(start, end):
            labels[i] = f"B-{entity}" if i == start else f"I-{entity}"
    return labels
