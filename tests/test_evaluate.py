from evaluate import _normalize, summarize


def test_normalize_ignores_case_and_surrounding_whitespace():
    assert _normalize("  Acme Co  ") == _normalize("acme co")


def test_summarize_marks_error_documents_as_all_incorrect():
    # evaluate_document() itself calls process_invoice, which needs a real
    # model — that path is covered by test_app.py. This targets the
    # summary logic against the shape evaluate_document() produces for a
    # document that failed to scan.
    results = [{
        "field": "Supplier Name", "expected": "Acme Co", "extracted": None,
        "confidence": None, "correct": False, "flagged": True, "reason": "document failed to scan",
    }]
    summary = summarize(results, type_matches=0, total_docs=1)
    assert summary["field_accuracy"] == 0.0
    assert summary["document_type_accuracy"] == 0.0


def test_summarize_computes_field_accuracy_and_confidence_calibration():
    results = [
        {"field": "A", "correct": True, "flagged": False},
        {"field": "A", "correct": True, "flagged": False},
        {"field": "B", "correct": False, "flagged": True},   # wrong AND correctly flagged
        {"field": "B", "correct": False, "flagged": False},  # wrong but NOT flagged — a real miss
        {"field": "C", "correct": True, "flagged": True},    # right but flagged — a false alarm
    ]
    summary = summarize(results, type_matches=2, total_docs=2)

    assert summary["field_accuracy"] == 3 / 5
    assert summary["document_type_accuracy"] == 1.0
    assert summary["wrong_fields_that_were_flagged"] == "1/2"
    assert summary["right_fields_that_were_not_flagged"] == "2/3"
    assert summary["per_field_accuracy"] == {"A": "2/2", "B": "0/2", "C": "1/1"}


def test_summarize_handles_zero_documents_without_dividing_by_zero():
    summary = summarize([], type_matches=0, total_docs=0)
    assert summary["field_accuracy"] == 0.0
    assert summary["document_type_accuracy"] == 0.0
    assert summary["wrong_fields_that_were_flagged"] == "0/0"
