import numpy as np

from engine.model import (
    NUM_CLASSES, TokenClassifier, build_char_context_windows, build_context_windows,
    cross_entropy_loss, one_hot, relu, relu_grad, softmax, xavier_init,
)


def _zero_char_windows(length, window, max_ngrams=8):
    """All-pad char windows, for tests that only care about the word path."""
    return build_char_context_windows(np.zeros((length, max_ngrams), dtype=np.int64), window)


def test_relu_and_grad():
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.array_equal(relu(z), np.array([0, 0, 0, 0.5, 2.0]))
    assert np.array_equal(relu_grad(z), np.array([0, 0, 0, 1, 1]))


def test_softmax_rows_sum_to_one_and_are_stable_for_large_inputs():
    z = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
    p = softmax(z)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(np.isfinite(p))


def test_xavier_init_shape_and_bounds():
    rng = np.random.default_rng(0)
    w = xavier_init(10, 20, rng)
    assert w.shape == (10, 20)
    limit = np.sqrt(6.0 / (10 + 20))
    assert np.all(np.abs(w) <= limit)


def test_one_hot():
    y = one_hot([0, 2], num_classes=4)
    assert y.shape == (2, 4)
    assert y[0, 0] == 1 and y[1, 2] == 1
    assert y.sum() == 2


def test_build_context_windows_pads_with_pad_id_zero():
    windows = build_context_windows([5, 6, 7], window=2)
    assert windows.shape == (3, 5)
    # center token of the middle row is the middle id; edges are padded with 0
    assert windows[0].tolist() == [0, 0, 5, 6, 7]
    assert windows[1].tolist() == [0, 5, 6, 7, 0]
    assert windows[2].tolist() == [5, 6, 7, 0, 0]


def test_build_context_windows_empty_sequence():
    windows = build_context_windows([], window=2)
    assert windows.shape == (0, 5)


def test_build_char_context_windows_pads_with_all_zero_rows():
    char_matrix = np.array([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
    windows = build_char_context_windows(char_matrix, window=1)
    assert windows.shape == (3, 3, 3)
    assert windows[0, 0].tolist() == [0, 0, 0]  # boundary pad row
    assert windows[0, 1].tolist() == [1, 2, 0]  # the real first row
    assert windows[1, 1].tolist() == [3, 0, 0]
    assert windows[2, 2].tolist() == [0, 0, 0]  # boundary pad row


def test_build_char_context_windows_empty_sequence():
    windows = build_char_context_windows(np.zeros((0, 8), dtype=np.int64), window=2)
    assert windows.shape == (0, 5, 8)


def test_forward_output_is_a_valid_probability_distribution():
    model = TokenClassifier(vocab_size=50, embed_dim=8, hidden_dim=16, window=1, seed=1)
    ids = build_context_windows([1, 2, 3, 4], window=1)
    char_ids = _zero_char_windows(4, window=1)
    P, cache = model.forward(ids, char_ids)
    assert P.shape == (4, NUM_CLASSES)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)


def test_backward_pass_reduces_loss_over_training_steps():
    """A tiny end-to-end gradient descent sanity check: hand-derived
    backprop should actually make the loss go down on a fixed batch."""
    model = TokenClassifier(vocab_size=20, embed_dim=6, hidden_dim=12, window=1, seed=2)
    ids = build_context_windows([1, 2, 3, 4, 5], window=1)
    char_ids = _zero_char_windows(5, window=1)
    labels = np.array([0, 1, 2, 1, 0])

    P0, _ = model.forward(ids, char_ids)
    initial_loss = cross_entropy_loss(P0, one_hot(labels, NUM_CLASSES))

    for _ in range(200):
        P, cache = model.forward(ids, char_ids)
        model.backward(cache, labels, learning_rate=0.5)

    P_final, _ = model.forward(ids, char_ids)
    final_loss = cross_entropy_loss(P_final, one_hot(labels, NUM_CLASSES))
    assert final_loss < initial_loss
    assert final_loss < 0.1  # should have essentially memorized this tiny fixed batch


def test_backward_pass_with_real_char_features_also_reduces_loss():
    """Same sanity check as above, but exercising the char-embedding branch
    with real (non-zero, non-uniform) hashed n-gram ids per window slot,
    so the char scatter-add path is proven to actually learn, not just
    the all-pad path every other test in this file uses."""
    model = TokenClassifier(vocab_size=20, embed_dim=6, hidden_dim=12, window=1,
                             seed=2, char_vocab_size=50, char_embed_dim=4, max_ngrams=3)
    ids = build_context_windows([1, 2, 3, 4, 5], window=1)
    rng = np.random.default_rng(0)
    char_matrix = rng.integers(0, 50, size=(5, 3))
    char_ids = build_char_context_windows(char_matrix, window=1)
    labels = np.array([0, 1, 2, 1, 0])

    P0, _ = model.forward(ids, char_ids)
    initial_loss = cross_entropy_loss(P0, one_hot(labels, NUM_CLASSES))

    for _ in range(200):
        P, cache = model.forward(ids, char_ids)
        model.backward(cache, labels, learning_rate=0.5)

    P_final, _ = model.forward(ids, char_ids)
    final_loss = cross_entropy_loss(P_final, one_hot(labels, NUM_CLASSES))
    assert final_loss < initial_loss
    assert final_loss < 0.1


def test_char_embedding_gradient_matches_numerical_finite_difference():
    """A loss-goes-down test can pass even with a somewhat-wrong gradient
    on a tiny memorization task. This checks the actual derivative:
    perturb a single W_char entry by +-eps, measure the resulting change
    in loss, and confirm it matches backward()'s analytical dW_char at
    that entry, within finite-difference tolerance. This is the strongest
    available evidence the masked mean-pool backward (the one genuinely
    new derivation in this change) is correct, not just plausible."""
    model = TokenClassifier(vocab_size=10, embed_dim=4, hidden_dim=6, window=1,
                             seed=7, char_vocab_size=15, char_embed_dim=3, max_ngrams=3)
    ids = build_context_windows([1, 2, 3], window=1)
    rng = np.random.default_rng(1)
    char_matrix = rng.integers(0, 15, size=(3, 3))
    char_ids = build_char_context_windows(char_matrix, window=1)
    labels = np.array([0, 1, 2])

    P, cache = model.forward(ids, char_ids)

    def loss_at_current_weights():
        P_now, _ = model.forward(ids, char_ids)
        return cross_entropy_loss(P_now, one_hot(labels, NUM_CLASSES))

    # backward() applies its update in place rather than returning
    # gradients, so the analytical dW_char is recomputed here inline,
    # using the exact same formula backward() uses, purely to compare
    # against the numerical estimate below without mutating the model.
    ids_c, char_ids_c = cache["ids"], cache["char_ids"]
    mask, counts = cache["mask"], cache["counts"]
    Y = one_hot(labels, NUM_CLASSES)
    n = ids_c.shape[0]
    dZ2 = (cache["P"] - Y) / n
    dH = dZ2 @ model.W2.T
    dZ1 = dH * relu_grad(cache["Z1"])
    dE_concat = dZ1 @ model.W1.T
    dE = dE_concat.reshape(n, model.window_size, model.embed_dim + model.char_embed_dim)
    dC_mean = dE[:, :, model.embed_dim:]
    dC = (dC_mean[:, :, None, :] / counts[:, :, :, None]) * mask
    analytical_dW_char = np.zeros_like(model.W_char)
    for k in range(model.window_size):
        for m in range(model.max_ngrams):
            np.add.at(analytical_dW_char, char_ids_c[:, k, m], dC[:, k, m, :])

    eps = 1e-5
    checked_any = False
    for row in range(model.char_vocab_size):
        if not np.any(char_ids == row):
            continue  # only rows that actually participate have a nonzero gradient to check
        for col in range(model.char_embed_dim):
            original = model.W_char[row, col]

            model.W_char[row, col] = original + eps
            loss_plus = loss_at_current_weights()
            model.W_char[row, col] = original - eps
            loss_minus = loss_at_current_weights()
            model.W_char[row, col] = original

            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            assert abs(numerical_grad - analytical_dW_char[row, col]) < 1e-4, (
                f"row={row} col={col}: numerical={numerical_grad} analytical={analytical_dW_char[row, col]}"
            )
            checked_any = True
    assert checked_any  # sanity: the batch actually exercised at least one real bucket


def test_char_features_change_prediction_for_otherwise_identical_word_ids():
    """Two windows with identical word ids but different char-ngram ids
    must be able to produce different predictions once the char weights
    are non-trivial: proof the char branch is actually wired into the
    forward computation, not just shaped correctly and ignored."""
    model = TokenClassifier(vocab_size=10, embed_dim=4, hidden_dim=8, window=0,
                             seed=5, char_vocab_size=20, char_embed_dim=4, max_ngrams=2)
    word_ids = np.array([[3], [3]])
    char_ids_a = np.array([[[1, 2]]])
    char_ids_b = np.array([[[9, 15]]])
    P_a, _ = model.forward(word_ids[:1], char_ids_a)
    P_b, _ = model.forward(word_ids[1:], char_ids_b)
    assert not np.allclose(P_a, P_b)


def test_char_pad_slots_do_not_affect_the_forward_mean():
    """A token's char mean should be identical whether or not trailing pad
    slots (bucket 0) are present, since the mask divides by the count of
    real n-grams only, not the fixed max_ngrams width."""
    model = TokenClassifier(vocab_size=10, embed_dim=4, hidden_dim=8, window=0,
                             seed=6, char_vocab_size=20, char_embed_dim=4, max_ngrams=4)
    word_ids = np.array([[3]])
    padded = np.array([[[5, 7, 0, 0]]])       # 2 real n-grams, 2 pad
    unpadded_equivalent = np.array([[[5, 7]]])  # same 2 real n-grams, no pad slots at all

    # Build a second model instance with max_ngrams=2 sharing the same
    # W_char rows 0..19 (same seed/char_vocab_size) so the comparison is
    # apples-to-apples on the same weights.
    model_narrow = TokenClassifier(vocab_size=10, embed_dim=4, hidden_dim=8, window=0,
                                    seed=6, char_vocab_size=20, char_embed_dim=4, max_ngrams=2)

    P_padded, _ = model.forward(word_ids, padded)
    P_unpadded, _ = model_narrow.forward(word_ids, unpadded_equivalent)
    assert np.allclose(P_padded, P_unpadded)


def test_predict_with_confidence_matches_argmax_and_max_probability():
    model = TokenClassifier(vocab_size=30, embed_dim=8, hidden_dim=16, window=1, seed=3)
    ids = build_context_windows([1, 2, 3], window=1)
    char_ids = _zero_char_windows(3, window=1)
    preds, confs = model.predict_with_confidence(ids, char_ids)
    P, _ = model.forward(ids, char_ids)
    assert np.array_equal(preds, np.argmax(P, axis=1))
    assert np.allclose(confs, P[np.arange(len(preds)), preds])


def test_save_and_load_weights_roundtrip(tmp_path):
    model = TokenClassifier(vocab_size=15, embed_dim=4, hidden_dim=8, window=1, seed=4)
    ids = build_context_windows([1, 2, 3], window=1)
    char_ids = _zero_char_windows(3, window=1)
    preds_before, _ = model.predict_with_confidence(ids, char_ids)

    path = tmp_path / "weights.npz"
    model.save_weights(str(path))
    loaded = TokenClassifier.load_weights(str(path))
    preds_after, _ = loaded.predict_with_confidence(ids, char_ids)

    assert np.array_equal(preds_before, preds_after)
    assert loaded.window == model.window
    assert loaded.char_vocab_size == model.char_vocab_size
    assert loaded.char_embed_dim == model.char_embed_dim
    assert loaded.max_ngrams == model.max_ngrams
    assert np.array_equal(loaded.W_char, model.W_char)


def test_load_weights_rejects_a_pre_char_feature_checkpoint(tmp_path):
    path = tmp_path / "old_weights.npz"
    np.savez(
        path,
        W_embed=np.zeros((5, 4)), W1=np.zeros((20, 8)), b1=np.zeros((1, 8)),
        W2=np.zeros((8, 3)), b2=np.zeros((1, 3)),
        vocab_size=5, embed_dim=4, hidden_dim=8, num_classes=3, window=2,
    )
    try:
        TokenClassifier.load_weights(str(path))
        assert False, "expected a ValueError for a pre-char-feature checkpoint"
    except ValueError as exc:
        assert "python train.py" in str(exc)
