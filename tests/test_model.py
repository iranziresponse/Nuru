import numpy as np

from engine.model import (
    NUM_CLASSES, TokenClassifier, build_context_windows, cross_entropy_loss,
    one_hot, relu, relu_grad, softmax, xavier_init,
)


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


def test_forward_output_is_a_valid_probability_distribution():
    model = TokenClassifier(vocab_size=50, embed_dim=8, hidden_dim=16, window=1, seed=1)
    ids = build_context_windows([1, 2, 3, 4], window=1)
    P, cache = model.forward(ids)
    assert P.shape == (4, NUM_CLASSES)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)


def test_backward_pass_reduces_loss_over_training_steps():
    """A tiny end-to-end gradient descent sanity check: hand-derived
    backprop should actually make the loss go down on a fixed batch."""
    model = TokenClassifier(vocab_size=20, embed_dim=6, hidden_dim=12, window=1, seed=2)
    ids = build_context_windows([1, 2, 3, 4, 5], window=1)
    labels = np.array([0, 1, 2, 1, 0])

    P0, _ = model.forward(ids)
    initial_loss = cross_entropy_loss(P0, one_hot(labels, NUM_CLASSES))

    for _ in range(200):
        P, cache = model.forward(ids)
        model.backward(cache, labels, learning_rate=0.5)

    P_final, _ = model.forward(ids)
    final_loss = cross_entropy_loss(P_final, one_hot(labels, NUM_CLASSES))
    assert final_loss < initial_loss
    assert final_loss < 0.1  # should have essentially memorized this tiny fixed batch


def test_predict_with_confidence_matches_argmax_and_max_probability():
    model = TokenClassifier(vocab_size=30, embed_dim=8, hidden_dim=16, window=1, seed=3)
    ids = build_context_windows([1, 2, 3], window=1)
    preds, confs = model.predict_with_confidence(ids)
    P, _ = model.forward(ids)
    assert np.array_equal(preds, np.argmax(P, axis=1))
    assert np.allclose(confs, P[np.arange(len(preds)), preds])


def test_save_and_load_weights_roundtrip(tmp_path):
    model = TokenClassifier(vocab_size=15, embed_dim=4, hidden_dim=8, window=1, seed=4)
    ids = build_context_windows([1, 2, 3], window=1)
    preds_before, _ = model.predict_with_confidence(ids)

    path = tmp_path / "weights.npz"
    model.save_weights(str(path))
    loaded = TokenClassifier.load_weights(str(path))
    preds_after, _ = loaded.predict_with_confidence(ids)

    assert np.array_equal(preds_before, preds_after)
    assert loaded.window == model.window
