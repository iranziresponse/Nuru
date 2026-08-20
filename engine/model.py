"""Pure matrix-multiplication forward/backward math for the token classifier.

No autograd, no framework layers: every derivative below is hand-derived
and hand-coded with NumPy as the only array library.

Architecture (per-token feed-forward tagger, layout-agnostic):

    Word embedding  : E_word = X_word . W_embed        (one-hot X, row lookup)
    Char embedding  : C      = mean over hashed character n-gram lookups
                                in W_char, masked so padding contributes
                                nothing (engine/tokenizer.py builds the
                                n-gram ids; this gives an unfamiliar word
                                a representation built from its spelling,
                                instead of collapsing to <UNK> like a
                                purely word-level vocabulary would)
    Context window  : E      = concat(E_word, C) per position, then
                                concat across the window -> E_concat
    Hidden Layer    : Z1 = E_concat . W1 + b1
                       H  = ReLU(Z1)
    Output Layer    : Z2 = H . W2 + b2                  (logits)
                       P  = softmax(Z2)
"""

import numpy as np

LABELS = [
    "O",
    "B-Vendor", "I-Vendor", "B-Date", "B-Total", "B-Tax",                    # Invoice
    "B-Merchant", "I-Merchant", "B-PaymentMethod", "I-PaymentMethod",        # Receipt
    "B-Account", "I-Account", "B-Period", "I-Period", "B-Balance",           # Statement
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)


def xavier_init(fan_in, fan_out, rng):
    """Glorot/Xavier uniform initialization: U(-limit, +limit)."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64)


def relu(z):
    return np.maximum(0.0, z)


def relu_grad(z):
    return (z > 0.0).astype(np.float64)


def softmax(z):
    """Row-wise, numerically stable softmax."""
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def cross_entropy_loss(P, Y_onehot):
    """Mean categorical cross-entropy L = -sum(Y * log(P)) over the batch."""
    eps = 1e-12
    per_token = -np.sum(Y_onehot * np.log(P + eps), axis=1)
    return float(np.mean(per_token))


def one_hot(label_ids, num_classes=NUM_CLASSES):
    n = len(label_ids)
    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n), label_ids] = 1.0
    return Y


def build_context_windows(id_sequence, window):
    """Turn one document's flat id sequence into (len(seq), 2*window+1) ids.

    A plain per-token classifier only ever sees a token's own embedding, so
    it can't use neighboring words (e.g. "Tax" before a dollar amount) to
    disambiguate. Sliding a small window over the sequence and concatenating
    the neighbors' embeddings gives the feed-forward network that local
    context, without introducing any recurrence. Boundaries are padded with
    id 0 (<PAD>), which never leaks across documents since this is called
    per-document.
    """
    ids = np.asarray(id_sequence, dtype=np.int64)
    length = len(ids)
    padded = np.concatenate([np.zeros(window, dtype=np.int64), ids, np.zeros(window, dtype=np.int64)])
    if length == 0:
        return np.zeros((0, 2 * window + 1), dtype=np.int64)
    return np.stack([padded[i:i + 2 * window + 1] for i in range(length)])


def build_char_context_windows(char_matrix, window):
    """Same sliding-window, same boundary contract as
    build_context_windows: char_matrix is one document's
    (length, max_ngrams) character-bucket ids (from
    engine.tokenizer.char_ngram_matrix); returns
    (length, 2*window+1, max_ngrams), padded with all-zero (all-pad) rows
    at document edges only.
    """
    char_matrix = np.asarray(char_matrix, dtype=np.int64)
    length, max_ngrams = char_matrix.shape
    pad_rows = np.zeros((window, max_ngrams), dtype=np.int64)
    padded = np.concatenate([pad_rows, char_matrix, pad_rows], axis=0)
    if length == 0:
        return np.zeros((0, 2 * window + 1, max_ngrams), dtype=np.int64)
    return np.stack([padded[i:i + 2 * window + 1] for i in range(length)])


class TokenClassifier:
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64,
                 num_classes=NUM_CLASSES, window=2, seed=42,
                 char_vocab_size=4001, char_embed_dim=16, max_ngrams=8):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.window = window
        self.window_size = 2 * window + 1
        # Defaults here mirror engine/tokenizer.py's CHAR_VOCAB_SIZE (4001,
        # i.e. CHAR_BUCKETS=4000 plus the reserved pad row) and
        # MAX_NGRAMS_PER_TOKEN (8). model.py deliberately doesn't import
        # tokenizer.py (each stays a leaf module app.py composes), so the
        # numbers are spelled out here rather than shared as a constant;
        # keep them in sync if tokenizer.py's ever change.
        self.char_vocab_size = char_vocab_size
        self.char_embed_dim = char_embed_dim
        self.max_ngrams = max_ngrams
        input_dim = self.window_size * (embed_dim + char_embed_dim)

        self.W_embed = xavier_init(vocab_size, embed_dim, rng)
        self.W_char = xavier_init(char_vocab_size, char_embed_dim, rng)
        self.W1 = xavier_init(input_dim, hidden_dim, rng)
        self.b1 = np.zeros((1, hidden_dim), dtype=np.float64)
        self.W2 = xavier_init(hidden_dim, num_classes, rng)
        self.b2 = np.zeros((1, num_classes), dtype=np.float64)

    # ---- forward pass ----------------------------------------------------
    def forward(self, window_ids, window_char_ids):
        """window_ids: (N, 2*window+1) word ids.
        window_char_ids: (N, 2*window+1, max_ngrams) hashed char-bucket ids,
        0 = pad (from engine.model.build_char_context_windows). Returns
        (P, cache)."""
        ids = np.asarray(window_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids.reshape(-1, 1)  # window=0 fallback: single-token input
        char_ids = np.asarray(window_char_ids, dtype=np.int64)
        n = ids.shape[0]

        E_word = self.W_embed[ids]                     # (N, K, D_word)

        C = self.W_char[char_ids]                       # (N, K, M, D_char)
        mask = (char_ids != 0).astype(np.float64)[..., None]  # (N, K, M, 1)
        counts = np.maximum(mask.sum(axis=2), 1.0)       # (N, K, 1): real n-grams per position, >=1 to avoid /0
        C_mean = (C * mask).sum(axis=2) / counts          # (N, K, D_char): mean over real n-grams only

        E = np.concatenate([E_word, C_mean], axis=-1)     # (N, K, D_word+D_char)
        E_concat = E.reshape(n, -1)                        # (N, K*(D_word+D_char))
        Z1 = E_concat @ self.W1 + self.b1                   # (N, H)
        H = relu(Z1)
        Z2 = H @ self.W2 + self.b2                           # (N, C)
        P = softmax(Z2)
        cache = {
            "ids": ids, "char_ids": char_ids, "mask": mask, "counts": counts,
            "E_concat": E_concat, "Z1": Z1, "H": H, "Z2": Z2, "P": P,
        }
        return P, cache

    # ---- backward pass (manual chain rule) --------------------------------
    def backward(self, cache, label_ids, learning_rate):
        ids, char_ids = cache["ids"], cache["char_ids"]
        mask, counts = cache["mask"], cache["counts"]
        E_concat, Z1, H, P = cache["E_concat"], cache["Z1"], cache["H"], cache["P"]
        n = ids.shape[0]
        Y = one_hot(label_ids, self.num_classes)

        # Output layer: dL/dZ2 simplifies to (P - Y) for softmax + cross-entropy.
        dZ2 = (P - Y) / n                            # (N, C)
        dW2 = H.T @ dZ2                              # (H, C)
        db2 = np.sum(dZ2, axis=0, keepdims=True)     # (1, C)

        # Hidden layer.
        dH = dZ2 @ self.W2.T                         # (N, H)
        dZ1 = dH * relu_grad(Z1)                      # (N, H)
        dW1 = E_concat.T @ dZ1                        # (K*(D_word+D_char), H)
        db1 = np.sum(dZ1, axis=0, keepdims=True)      # (1, H)

        # Split the concatenated-input gradient back into its word and char
        # halves, mirroring how forward() concatenated them.
        dE_concat = dZ1 @ self.W1.T                   # (N, K*(D_word+D_char))
        dE = dE_concat.reshape(n, self.window_size, self.embed_dim + self.char_embed_dim)
        dE_word = dE[:, :, :self.embed_dim]            # (N, K, D_word)
        dC_mean = dE[:, :, self.embed_dim:]             # (N, K, D_char)

        # Word embedding layer: scatter-add since X is one-hot (sparse row
        # lookup), once per window slot so a token gets gradient from every
        # position its embedding was concatenated into.
        dW_embed = np.zeros_like(self.W_embed)
        for k in range(self.window_size):
            np.add.at(dW_embed, ids[:, k], dE_word[:, k, :])

        # Char embedding layer: backward of a masked mean-pool. Each real
        # n-gram slot gets an equal share (1/count) of dC_mean; pad slots
        # get none, since the mask already excluded them from the forward
        # mean, so their gradient must be zero too.
        dC = (dC_mean[:, :, None, :] / counts[:, :, :, None]) * mask  # (N, K, M, D_char)
        dW_char = np.zeros_like(self.W_char)
        for k in range(self.window_size):
            for m in range(self.max_ngrams):
                np.add.at(dW_char, char_ids[:, k, m], dC[:, k, m, :])

        # Gradient descent update: W_new = W_old - alpha * dL/dW
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W_embed -= learning_rate * dW_embed
        self.W_char -= learning_rate * dW_char

    # ---- inference ----------------------------------------------------
    def predict(self, window_ids, window_char_ids):
        P, _ = self.forward(window_ids, window_char_ids)
        return np.argmax(P, axis=1)

    def predict_with_confidence(self, window_ids, window_char_ids):
        """Returns (pred_ids, confidences). Confidence is the softmax
        probability the model assigned to its own predicted class, straight
        from the P = softmax(Z) output layer."""
        P, _ = self.forward(window_ids, window_char_ids)
        preds = np.argmax(P, axis=1)
        confidences = P[np.arange(len(preds)), preds]
        return preds, confidences

    # ---- persistence ----------------------------------------------------
    def save_weights(self, path):
        np.savez(
            path,
            W_embed=self.W_embed, W_char=self.W_char, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            vocab_size=self.vocab_size, embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim, num_classes=self.num_classes,
            window=self.window, char_vocab_size=self.char_vocab_size,
            char_embed_dim=self.char_embed_dim, max_ngrams=self.max_ngrams,
        )

    @classmethod
    def load_weights(cls, path):
        data = np.load(path)
        if "W_char" not in data:
            raise ValueError(
                "This checkpoint predates character-level features and can't "
                "be loaded. Run `python train.py` to regenerate "
                "engine/weights.npz and engine/vocab.json."
            )
        model = cls(
            vocab_size=int(data["vocab_size"]),
            embed_dim=int(data["embed_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            num_classes=int(data["num_classes"]),
            window=int(data["window"]) if "window" in data else 2,
            char_vocab_size=int(data["char_vocab_size"]),
            char_embed_dim=int(data["char_embed_dim"]),
            max_ngrams=int(data["max_ngrams"]),
        )
        model.W_embed = data["W_embed"]
        model.W_char = data["W_char"]
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        return model
