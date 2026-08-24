from engine.tokenizer import tokenize

from data.align_real_labels import (
    build_bio_labels, find_best_text_span, find_date_span, find_money_span,
)


# ---- find_best_text_span: name-like fields, using the actual garbled OCR
# text observed against the real SROIE benchmark, not idealized input ----

def test_finds_exact_company_name_when_ocr_is_clean():
    tokens = tokenize("BOOK TA K TAMAN DAYA SDN BHD")
    span = find_best_text_span(tokens, "BOOK TA .K (TAMAN DAYA) SDN BHD")
    assert span is not None
    start, end, ratio = span
    assert "".join(tokens[start:end]).replace(" ", "") != ""
    assert ratio > 0.85


def test_finds_company_name_through_real_ocr_garbling():
    """SDN BHD -> SDN SHD is a real observed misread (B->S); the
    normalized text still overlaps enough to clear the threshold."""
    tokens = tokenize("BOOK TA -K TAMAN DAYA SDN SHD")
    span = find_best_text_span(tokens, "BOOK TA .K (TAMAN DAYA) SDN BHD")
    assert span is not None
    _, _, ratio = span
    assert ratio >= 0.55


def test_finds_company_name_through_severe_garbling():
    """MR D.I.Y. -> MR O.1. is a real observed misread (a whole word
    corrupted, not just one letter). Confirms the fuzzy matcher survives
    more than a single-character typo, since real OCR noise is rarely
    that mild."""
    tokens = tokenize("MR O.1. JOHOR SDN BHD")
    span = find_best_text_span(tokens, "MR D.I.Y. (JOHOR) SDN BHD")
    assert span is not None


def test_returns_none_for_completely_unrelated_text():
    tokens = tokenize("THANK YOU PLEASE COME AGAIN")
    span = find_best_text_span(tokens, "BOOK TA .K (TAMAN DAYA) SDN BHD")
    assert span is None


def test_returns_none_for_empty_target():
    tokens = tokenize("SOME RECEIPT TEXT")
    assert find_best_text_span(tokens, "") is None


def test_prefers_the_best_matching_span_not_the_first_candidate():
    tokens = tokenize("tan woon yann BOOK TA K SDN BHD Total")
    span = find_best_text_span(tokens, "BOOK TA K SDN BHD")
    assert span is not None
    start, end, _ = span
    matched_text = "".join(tokens[start:end])
    assert "BOOK" in matched_text
    assert "tan" not in matched_text.lower() or "woon" not in matched_text.lower()


# ---- find_date_span: only real <DATE>-shaped tokens, digit-exact -----------

def test_finds_iso_date_exact_match():
    tokens = tokenize("Date 25/12/2018 Total")
    span = find_date_span(tokens, "25/12/2018")
    assert span == (1, 2)


def test_finds_dash_formatted_date_against_slash_ground_truth():
    tokens = tokenize("Date 19-03-2018 Cashier")
    span = find_date_span(tokens, "19-03-2018")
    assert span is not None


def test_finds_two_digit_year_date_against_four_digit_ground_truth():
    tokens = tokenize("Date 12-01-19 Total")
    span = find_date_span(tokens, "12-01-2019")
    assert span is not None


def test_does_not_match_an_unrelated_date():
    tokens = tokenize("Date 25/12/2018 Total")
    span = find_date_span(tokens, "01/01/2020")
    assert span is None


def test_ignores_non_date_shaped_tokens_even_with_matching_digits():
    """A bare number that happens to share digits with the target date
    but was never tokenized as a <DATE> shape must not be treated as one
    (e.g. an invoice number or quantity)."""
    tokens = tokenize("Qty 25122018 units")
    span = find_date_span(tokens, "25/12/2018")
    assert span is None


# ---- find_money_span: only real <MONEY>-shaped tokens, numeric-exact ------

def test_finds_exact_money_match():
    tokens = tokenize("Total 9.00 CASH")
    span = find_money_span(tokens, "9.00")
    assert span == (1, 2)


def test_finds_dollar_prefixed_money_match():
    tokens = tokenize("Total $9.00 CASH")
    span = find_money_span(tokens, "9.00")
    assert span is not None


def test_does_not_match_a_different_amount():
    tokens = tokenize("Subtotal 1.00 Total 9.00")
    span = find_money_span(tokens, "9.00")
    start, end = span
    assert tokens[start:end] == ["9.00"]


def test_returns_none_when_no_money_shaped_token_matches():
    tokens = tokenize("Total 12.50 CASH")
    assert find_money_span(tokens, "9.00") is None


# ---- build_bio_labels -------------------------------------------------------

def test_build_bio_labels_tags_multi_token_span_correctly():
    tokens = ["BOOK", "TA", "SDN", "BHD", "Total", "9.00"]
    labels = build_bio_labels(tokens, {"Merchant": (0, 4), "Total": (5, 6)})
    assert labels == ["B-Merchant", "I-Merchant", "I-Merchant", "I-Merchant", "O", "B-Total"]


def test_build_bio_labels_handles_missing_spans_gracefully():
    tokens = ["Thank", "you"]
    labels = build_bio_labels(tokens, {"Merchant": None, "Total": None})
    assert labels == ["O", "O"]


def test_build_bio_labels_single_token_span():
    tokens = ["Date", "25/12/2018"]
    labels = build_bio_labels(tokens, {"Date": (1, 2)})
    assert labels == ["O", "B-Date"]
