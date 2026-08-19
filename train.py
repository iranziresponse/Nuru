"""Executes the epoch loop and backpropagation for the token classifier.

Pipeline: [Forward Pass] -> [Cross-Entropy Loss] -> [Gradients] -> [Weight Update]

Usage:
    python train.py --epochs 40 --lr 0.5
"""

import argparse
import csv
import os

import numpy as np

from engine.model import LABEL2ID, TokenClassifier, build_context_windows, cross_entropy_loss, one_hot
from engine.tokenizer import Vocabulary

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset.csv")
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine", "weights.npz")
VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine", "vocab.json")


def load_documents(csv_path):
    """Group the token-per-row CSV back into per-document (tokens, labels)."""
    docs = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["doc_id"]
            docs.setdefault(doc_id, {"tokens": [], "labels": []})
            docs[doc_id]["tokens"].append(row["token"])
            docs[doc_id]["labels"].append(row["label"])
    return [(d["tokens"], d["labels"]) for d in docs.values()]


def split_train_val(docs, val_fraction, rng):
    indices = np.arange(len(docs))
    rng.shuffle(indices)
    n_val = max(1, int(len(docs) * val_fraction))
    val_idx = set(indices[:n_val].tolist())
    train_docs = [docs[i] for i in range(len(docs)) if i not in val_idx]
    val_docs = [docs[i] for i in range(len(docs)) if i in val_idx]
    return train_docs, val_docs


def flatten(docs, vocab, window):
    """Encode every document, build its context windows, and stack across docs.

    Windows are built per document (not on a global flattened id stream) so
    padding at a window's edge never bleeds context from an unrelated doc.
    """
    all_windows, all_labels = [], []
    for tokens, labels in docs:
        ids = vocab.encode(tokens)
        all_windows.append(build_context_windows(ids, window))
        all_labels.extend(LABEL2ID[l] for l in labels)
    X = np.concatenate(all_windows, axis=0) if all_windows else np.zeros((0, 2 * window + 1), dtype=np.int64)
    return X, np.array(all_labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Train the token classification network.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=2, help="Context tokens on each side.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading dataset from {DATA_PATH}")
    docs = load_documents(DATA_PATH)
    train_docs, val_docs = split_train_val(docs, args.val_fraction, rng)
    print(f"{len(docs)} documents -> {len(train_docs)} train / {len(val_docs)} val")

    vocab = Vocabulary()
    vocab.build([tokens for tokens, _ in train_docs])
    print(f"Vocabulary size: {vocab.size}")

    X_train, Y_train = flatten(train_docs, vocab, args.window)
    X_val, Y_val = flatten(val_docs, vocab, args.window)
    print(f"Train tokens: {len(X_train)}  Val tokens: {len(X_val)}")

    model = TokenClassifier(
        vocab_size=vocab.size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        window=args.window,
        seed=args.seed,
    )

    n = len(X_train)
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        X_shuf, Y_shuf = X_train[perm], Y_train[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, args.batch_size):
            end = start + args.batch_size
            batch_ids = X_shuf[start:end]
            batch_labels = Y_shuf[start:end]

            P, cache = model.forward(batch_ids)
            epoch_loss += cross_entropy_loss(P, one_hot(batch_labels))
            n_batches += 1

            model.backward(cache, batch_labels, learning_rate=args.lr)

        train_loss = epoch_loss / n_batches
        val_P, _ = model.forward(X_val)
        val_preds = np.argmax(val_P, axis=1)
        val_acc = float(np.mean(val_preds == Y_val))

        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    model.save_weights(WEIGHTS_PATH)
    vocab.save(VOCAB_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")
    print(f"Saved vocabulary to {VOCAB_PATH}")


if __name__ == "__main__":
    main()
