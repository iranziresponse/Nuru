"""String-to-integer vector mapping utilities.

No external NLP libraries: tokenization is a hand-written regex scanner,
and the vocabulary is a plain Python dict mapping token -> integer id.
"""

import json
import re

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
