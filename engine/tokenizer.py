"""String-to-integer vector mapping utilities.

No external NLP libraries: tokenization is a hand-written regex scanner,
and the vocabulary is a plain Python dict mapping token -> integer id.
"""

import json
import re

import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Order matters: more specific patterns (dates, money, percentages) are
# tried before generic word/number patterns so entities like "2026-08-15"
# or "$1,204.50" survive as a single token instead of being fragmented.
_TOKEN_PATTERN = re.compile(
    r"""
    \d{4}-\d{2}-\d{2}                     # ISO date: 2026-08-15
    |\d{1,2}/\d{1,2}/\d{2,4}              # US date: 08/15/2026
    |\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?     # money: $1,204.50
    |\d{1,3}(?:,\d{3})*\.\d{2}            # bare decimal amount: 1204.50
    |\d+(?:\.\d+)?%                       # percentage: 8.5%
    |[A-Za-z][A-Za-z&.'-]*                # word-like token
    |\d+                                  # bare integer
    |[^\s\w]                              # single punctuation character
    """,
    re.VERBOSE,
)


def tokenize(text):
    """Split raw text into a flat list of token strings."""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)


_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})$")
_MONEY_RE = re.compile(r"^(\$\d{1,3}(,\d{3})*(\.\d{2})?|\d{1,3}(,\d{3})*\.\d{2})$")
_PERCENT_RE = re.compile(r"^\d+(\.\d+)?%$")
_NUM_RE = re.compile(r"^\d+$")


def normalize(token):
    """Map a raw token to its vocabulary key.

    Dates, money amounts, percentages and bare integers are open-vocabulary:
    every invoice has different literal values, so a fresh invoice would
    otherwise mint tokens the model never saw during training and can only
    embed as <UNK> noise. Collapsing each shape to one shared pseudo-token
    lets the network learn "this position holds a date/amount", which
    generalizes to unseen numbers, while ordinary words keep their own
    embeddings so lexical context (e.g. "Tax", "Total", "Due") still guides
    the surrounding predictions.
    """
    if _DATE_RE.match(token):
        return "<DATE>"
    if _MONEY_RE.match(token):
        return "<MONEY>"
    if _PERCENT_RE.match(token):
        return "<PERCENT>"
    if _NUM_RE.match(token):
        return "<NUM>"
    return token.lower()


_SPECIAL_TOKENS = {"<DATE>", "<MONEY>", "<PERCENT>", "<NUM>"}

# A word-level vocabulary alone collapses every unfamiliar name to the same
# <UNK> id, giving the classifier no way to tell "Brightwater Consulting"
# from "Zylophant Nonsense": both look identical once encoded. Hashed
# character n-grams give every word, seen or not, a representation built
# from its spelling (capitalization patterns, suffixes like "Inc"/"Corp",
# name-ish endings), which is exactly the signal an OOV vendor/merchant
# name still carries even though its exact spelling was never in training.
# This is fastText's *hashing trick*: buckets are trained from scratch
# (Xavier-init in engine/model.py, same as the word embeddings), nothing
# pretrained is involved.
NGRAM_SIZES = (3, 4)
CHAR_BUCKETS = 4000          # bucket 0 is reserved for padding
CHAR_VOCAB_SIZE = CHAR_BUCKETS + 1
MAX_NGRAMS_PER_TOKEN = 8


def _fnv1a(s):
    """32-bit FNV-1a. Deliberately not Python's builtin hash(): that's
    randomized per-process (PYTHONHASHSEED) unless explicitly disabled,
    which would make a saved weights.npz checkpoint's char-bucket mapping
    different on every machine and every restart."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _char_ngrams(word):
    """Character n-grams of a word, wrapped with boundary markers so a
    prefix/suffix n-gram is distinguishable from the same substring
    appearing mid-word (e.g. the leading "Con" of "Consulting" vs. a
    mid-word "con"). Deduplicated preserving first-occurrence order; if
    more than MAX_NGRAMS_PER_TOKEN survive, stride-subsampled across the
    full list rather than truncated to a prefix, so a long word's suffix
    (often the most distinctive part, e.g. "-berg", "-son", "LLC") still
    contributes instead of only ever seeing its first few characters."""
    wrapped = f"<{word.lower()}>"
    seen = []
    seen_set = set()
    for n in NGRAM_SIZES:
        if len(wrapped) < n:
            continue
        for i in range(len(wrapped) - n + 1):
            ngram = wrapped[i:i + n]
            if ngram not in seen_set:
                seen_set.add(ngram)
                seen.append(ngram)
    if len(seen) > MAX_NGRAMS_PER_TOKEN:
        stride = max(1, len(seen) // MAX_NGRAMS_PER_TOKEN)
        seen = seen[::stride][:MAX_NGRAMS_PER_TOKEN]
    return seen


def char_ngram_ids(token):
    """MAX_NGRAMS_PER_TOKEN hashed bucket ids for one raw token, zero
    (pad) filled past however many n-grams it actually has. The <DATE>/
    <MONEY>/<PERCENT>/<NUM> pseudo-tokens get an all-pad vector: their
    literal digit text carries no reusable subword signal and would just
    add hash noise across otherwise-unrelated amounts and dates. Every
    ordinary word gets real char features, including one that will still
    map to <UNK> at the word level, since that's exactly the case this
    exists to help."""
    ids = [0] * MAX_NGRAMS_PER_TOKEN
    if normalize(token) in _SPECIAL_TOKENS:
        return ids
    for i, ngram in enumerate(_char_ngrams(token)):
        ids[i] = 1 + (_fnv1a(ngram) % CHAR_BUCKETS)
    return ids


def char_ngram_matrix(tokens):
    """(len(tokens), MAX_NGRAMS_PER_TOKEN) int array, one row per token."""
    if not tokens:
        return np.zeros((0, MAX_NGRAMS_PER_TOKEN), dtype=np.int64)
    return np.array([char_ngram_ids(tok) for tok in tokens], dtype=np.int64)


class Vocabulary:
    """Static word/token -> integer index map built from a training corpus."""

    def __init__(self):
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, token_lists):
        """Pass over the dataset, collect every unique token, assign an id."""
        for tokens in token_lists:
            for tok in tokens:
                key = normalize(tok)
                if key not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[key] = idx
                    self.idx2word[idx] = key
        return self

    @property
    def size(self):
        return len(self.word2idx)

    def encode(self, tokens):
        unk = self.word2idx[UNK_TOKEN]
        return [self.word2idx.get(normalize(tok), unk) for tok in tokens]

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path, "r", encoding="utf-8") as f:
            vocab.word2idx = json.load(f)
        vocab.idx2word = {v: k for k, v in vocab.word2idx.items()}
        return vocab


class Tokenizer:
    """Wraps tokenize() + Vocabulary into a single string -> id-array utility."""

    def __init__(self, vocab=None):
        self.vocab = vocab or Vocabulary()

    def fit(self, texts):
        token_lists = [tokenize(t) for t in texts]
        self.vocab.build(token_lists)
        return token_lists

    def encode_text(self, text):
        tokens = tokenize(text)
        return tokens, self.vocab.encode(tokens)

    def save(self, path):
        self.vocab.save(path)

    @classmethod
    def load(cls, path):
        return cls(vocab=Vocabulary.load(path))
