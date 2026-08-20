from engine.tokenizer import (
    CHAR_BUCKETS, MAX_NGRAMS_PER_TOKEN, Vocabulary,
    _char_ngrams, _fnv1a, char_ngram_ids, char_ngram_matrix, normalize, tokenize,
)


def test_tokenize_splits_words_dates_and_money():
    tokens = tokenize("Invoice Date: 2026-08-15 Total: $1,204.50")
    assert tokens == ["Invoice", "Date", ":", "2026-08-15", "Total", ":", "$1,204.50"]


def test_tokenize_handles_percent_and_us_date():
    tokens = tokenize("Tax (8.5%) due 08/15/2026")
    assert "8.5%" in tokens
    assert "08/15/2026" in tokens


def test_tokenize_empty_input():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_normalize_collapses_open_vocabulary_shapes():
    assert normalize("2026-08-15") == "<DATE>"
    assert normalize("08/15/2026") == "<DATE>"
    assert normalize("$1,204.50") == "<MONEY>"
    assert normalize("8.5%") == "<PERCENT>"
    assert normalize("4821") == "<NUM>"
    assert normalize("Invoice") == "invoice"  # ordinary word: lowercased, not collapsed


def test_vocabulary_build_and_encode_roundtrip():
    vocab = Vocabulary()
    vocab.build([["Total", ":", "$500.00"], ["Vendor", "Acme"]])
    ids = vocab.encode(["Total", ":", "$999.99"])  # different amount, same <MONEY> bucket
    assert vocab.encode(["Total"])[0] == vocab.encode(["total"])[0]  # case-insensitive
    assert ids[2] == vocab.encode(["$500.00"])[0]  # both money values collapse to the same id


def test_vocabulary_unknown_token_maps_to_unk():
    vocab = Vocabulary()
    vocab.build([["hello"]])
    unk_id = vocab.word2idx["<UNK>"]
    assert vocab.encode(["totally-unseen-word-xyz"])[0] == unk_id


# ---- character n-gram hashing (OOV word representation) ---------------------

def test_fnv1a_is_deterministic_across_calls():
    assert _fnv1a("brightwater") == _fnv1a("brightwater")
    assert _fnv1a("brightwater") != _fnv1a("zylophant")


def test_char_ngrams_wraps_with_boundary_markers_and_dedupes():
    ngrams = _char_ngrams("ab")
    assert ngrams  # even a 2-char word yields at least one 3-gram: "<ab>"
    assert len(ngrams) == len(set(ngrams))  # deduplicated


def test_char_ngrams_caps_at_max_and_keeps_prefix_and_suffix_signal():
    ngrams = _char_ngrams("internationalconglomerate")
    assert len(ngrams) <= MAX_NGRAMS_PER_TOKEN
    # stride-subsampling, not prefix truncation, so the first surviving
    # n-gram starts at the wrapped word's boundary and the rest aren't
    # all clustered at the very front of the word.
    assert ngrams[0].startswith("<")


def test_char_ngram_ids_returns_fixed_length_bucket_ids_in_range():
    ids = char_ngram_ids("Brightwater")
    assert len(ids) == MAX_NGRAMS_PER_TOKEN
    for bucket_id in ids:
        assert 0 <= bucket_id <= CHAR_BUCKETS


def test_char_ngram_ids_is_all_pad_for_special_pseudo_tokens():
    for token in ("2026-08-15", "$1,204.50", "8.5%", "4821"):
        assert char_ngram_ids(token) == [0] * MAX_NGRAMS_PER_TOKEN


def test_char_ngram_ids_is_deterministic_and_case_insensitive():
    assert char_ngram_ids("Brightwater") == char_ngram_ids("brightwater")
    assert char_ngram_ids("Brightwater") == char_ngram_ids("Brightwater")


def test_char_ngram_ids_shares_buckets_for_similarly_suffixed_unfamiliar_words():
    """The whole point of the mechanism: two unseen words with real
    spelling overlap should share at least one hashed bucket, while two
    unrelated words normally shouldn't."""
    similar_a = set(char_ngram_ids("Brightwater"))
    similar_b = set(char_ngram_ids("Brightstone"))
    unrelated = set(char_ngram_ids("Zylophant"))
    shared_similar = (similar_a & similar_b) - {0}
    shared_unrelated = (similar_a & unrelated) - {0}
    assert shared_similar
    assert len(shared_similar) >= len(shared_unrelated)


def test_char_ngram_matrix_shape_and_empty_input():
    matrix = char_ngram_matrix(["Acme", "Corp"])
    assert matrix.shape == (2, MAX_NGRAMS_PER_TOKEN)
    empty = char_ngram_matrix([])
    assert empty.shape == (0, MAX_NGRAMS_PER_TOKEN)
