from engine.tokenizer import Vocabulary, normalize, tokenize


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
