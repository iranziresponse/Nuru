import pytest

from app import (
    TYPE_FIELD_CONFIG, VOCAB_PATH, WEIGHTS_PATH,
    decode_entities, infer_document_type, process_invoice,
)


def test_infer_document_type_priority():
    assert infer_document_type({"Vendor", "Tax"}) == "Invoice"
    assert infer_document_type({"Merchant"}) == "Receipt"
    assert infer_document_type({"PaymentMethod"}) == "Receipt"
    assert infer_document_type({"Account", "Balance"}) == "Statement"
    assert infer_document_type(set()) == "Invoice"  # fallback


def test_type_field_config_covers_every_document_kind():
    assert set(TYPE_FIELD_CONFIG.keys()) == {"Invoice", "Receipt", "Statement"}
    for type_name, fields in TYPE_FIELD_CONFIG.items():
        assert fields, f"{type_name} has no configured fields"
        keys = [k for k, _ in fields]
        assert len(keys) == len(set(keys)), f"{type_name} has duplicate entity keys"


def test_decode_entities_single_token_span():
    tokens = ["Total", ":", "$500.00"]
    labels = ["O", "O", "B-Total"]
    confs = [0.9, 0.9, 0.95]
    values, confidences = decode_entities(tokens, labels, confs)
    assert values == {"Total": ["$500.00"]}
    assert confidences == {"Total": [0.95]}


def test_decode_entities_multi_token_span_uses_min_confidence():
    tokens = ["Acme", "Supply", "Co", "Invoice"]
    labels = ["B-Vendor", "I-Vendor", "I-Vendor", "O"]
    confs = [0.99, 0.80, 0.95, 0.5]
    values, confidences = decode_entities(tokens, labels, confs)
    assert values == {"Vendor": ["Acme Supply Co"]}
    assert confidences["Vendor"] == [0.80]  # weakest token in the span


def test_decode_entities_orphan_continuation_tag_does_not_attach_to_stale_span():
    # A B-Vendor span is flushed by an O, then a later I-Vendor with no
    # matching B- shouldn't silently reattach to the old span's leftovers.
    tokens = ["Acme", "and", "Co"]
    labels = ["B-Vendor", "O", "I-Vendor"]
    confs = [0.9, 0.9, 0.9]
    values, _ = decode_entities(tokens, labels, confs)
    assert values == {"Vendor": ["Acme"]}  # "Co" must NOT be silently appended


def test_decode_entities_no_entities_found():
    values, confidences = decode_entities(["hello", "world"], ["O", "O"], [0.9, 0.9])
    assert values == {}
    assert confidences == {}


# ---- end-to-end extraction against a real generated PDF, using the
# repo's actual trained weights (checked in) ----------------------------

reportlab_canvas = pytest.importorskip("reportlab.pdfgen.canvas")


def _make_pdf(path, lines):
    c = reportlab_canvas.Canvas(str(path))
    y = 750
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()


@pytest.fixture(scope="module")
def model_and_vocab():
    import os
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(VOCAB_PATH):
        pytest.skip("model not trained; run train.py first")
    from engine.model import TokenClassifier
    from engine.tokenizer import Vocabulary
    return TokenClassifier.load_weights(WEIGHTS_PATH), Vocabulary.load(VOCAB_PATH)


def test_process_invoice_extracts_invoice_fields(tmp_path, model_and_vocab):
    model, vocab = model_and_vocab
    pdf_path = tmp_path / "invoice.pdf"
    _make_pdf(pdf_path, [
        "Quantum Electronics", "Invoice Number: INV-88213", "Invoice Date: 2026-04-02",
        "Bill To: Customer Account # 5521", "Subtotal: $1,240.00", "Tax (8.0%): $99.20",
        "Total Due: $1,339.20", "Payment is due within 30 days. Thank you for your business.",
    ])
    result = process_invoice(model, vocab, str(pdf_path))
    assert result["Error"] is False
    assert result["Type"] == "Invoice"
    values = {f["label"]: f["value"] for f in result["Fields"]}
    assert values["Supplier Name"] == "Quantum Electronics"
    assert values["Total Amount Due"] == "$1,339.20"


def test_process_invoice_handles_corrupt_pdf_without_raising(tmp_path, model_and_vocab):
    model, vocab = model_and_vocab
    bad_path = tmp_path / "not_a_pdf.pdf"
    bad_path.write_bytes(b"this is not a real pdf")
    result = process_invoice(model, vocab, str(bad_path))
    assert result["Error"] is True
    assert result["Fields"] == []
    assert result["NeedsReview"] is True
