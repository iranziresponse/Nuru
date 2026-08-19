"""Measures extraction accuracy against a fixed evaluation set, not just
loss/val-accuracy on the training distribution.

Reports, per field and overall: how often the extracted value exactly
matches ground truth, and whether the model's own confidence score is
actually informative, i.e. do wrong answers get flagged as low-confidence
more often than right ones do. That second question matters more than raw
accuracy for a review-before-send tool: a model that's occasionally wrong
but honest about it is more useful than one that's more often right but
never says so.

IMPORTANT: eval/documents/*.pdf are synthetic (see
eval/generate_eval_set.py), deliberately built with names and phrasing
absent from the training generator, so this is a genuine generalization
check, not the model grading its own homework. It is NOT a substitute for
benchmarking against real invoices, which this project does not have
access to. Point this script at a directory of real documents (with a
matching ground_truth.json) the moment any exist.

Usage:
    python evaluate.py
    python evaluate.py --documents path/to/real/pdfs --ground-truth path/to/truth.json
"""

import argparse
import json
import os

from app import DEFAULT_CONFIDENCE_THRESHOLD, VOCAB_PATH, WEIGHTS_PATH, process_invoice
from engine.model import TokenClassifier
from engine.tokenizer import Vocabulary

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval")
DEFAULT_DOCUMENTS_DIR = os.path.join(EVAL_DIR, "documents")
DEFAULT_GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.json")


def _normalize(value):
    return (value or "").strip().casefold()


def evaluate_document(model, vocab, pdf_path, expected, confidence_threshold):
    """Returns a list of per-field result dicts for one document."""
    result = process_invoice(model, vocab, pdf_path, confidence_threshold)
    results = []

    if result["Error"]:
        for label in expected["fields"]:
            results.append({
                "field": label, "expected": expected["fields"][label], "extracted": None,
                "confidence": None, "correct": False, "flagged": True, "reason": "document failed to scan",
            })
        return results, result["Type"]

    extracted_by_label = {f["label"]: f for f in result["Fields"]}
    for label, expected_value in expected["fields"].items():
        field = extracted_by_label.get(label)
        extracted_value = field["value"] if field else None
        confidence = field["confidence"] if field else None
        correct = _normalize(extracted_value) == _normalize(expected_value)
        flagged = (confidence is not None and confidence < confidence_threshold) or not extracted_value
        results.append({
            "field": label, "expected": expected_value, "extracted": extracted_value,
            "confidence": confidence, "correct": correct, "flagged": flagged, "reason": None,
        })
    return results, result["Type"]


def summarize(all_results, type_matches, total_docs):
    total_fields = len(all_results)
    correct_fields = sum(1 for r in all_results if r["correct"])

    wrong = [r for r in all_results if not r["correct"]]
    right = [r for r in all_results if r["correct"]]
    wrong_and_flagged = sum(1 for r in wrong if r["flagged"])
    right_and_not_flagged = sum(1 for r in right if not r["flagged"])

    per_field = {}
    for r in all_results:
        bucket = per_field.setdefault(r["field"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        if r["correct"]:
            bucket["correct"] += 1

    return {
        "documents": total_docs,
        "document_type_accuracy": type_matches / total_docs if total_docs else 0.0,
        "field_accuracy": correct_fields / total_fields if total_fields else 0.0,
        "total_fields": total_fields,
        "correct_fields": correct_fields,
        "wrong_fields_that_were_flagged": f"{wrong_and_flagged}/{len(wrong)}" if wrong else "0/0",
        "right_fields_that_were_not_flagged": f"{right_and_not_flagged}/{len(right)}" if right else "0/0",
        "per_field_accuracy": {
            label: f"{b['correct']}/{b['total']}" for label, b in sorted(per_field.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Measure Nuru's extraction accuracy against a labeled document set.")
    parser.add_argument("--documents", default=DEFAULT_DOCUMENTS_DIR)
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--report", default=None, help="Optional path to write the full JSON report.")
    args = parser.parse_args()

    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(VOCAB_PATH):
        raise SystemExit("No trained model found. Run `python train.py` first.")

    with open(args.ground_truth, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    vocab = Vocabulary.load(VOCAB_PATH)
    model = TokenClassifier.load_weights(WEIGHTS_PATH)

    all_results = []
    type_matches = 0
    failures = []

    for filename, expected in ground_truth.items():
        pdf_path = os.path.join(args.documents, filename)
        if not os.path.exists(pdf_path):
            print(f"skip (not found): {pdf_path}")
            continue
        results, inferred_type = evaluate_document(model, vocab, pdf_path, expected, args.confidence_threshold)
        all_results.extend(results)
        if inferred_type == expected["type"]:
            type_matches += 1
        else:
            failures.append(f"{filename}: expected type {expected['type']!r}, got {inferred_type!r}")
        for r in results:
            if not r["correct"]:
                failures.append(
                    f"{filename}: {r['field']!r} expected {r['expected']!r}, got {r['extracted']!r} "
                    f"(confidence={r['confidence']})"
                )

    summary = summarize(all_results, type_matches, len(ground_truth))

    print(f"\n{'=' * 60}")
    print("NURU EXTRACTION ACCURACY: SYNTHETIC HELD-OUT SET")
    print("(not a real-world accuracy claim, see evaluate.py docstring)")
    print(f"{'=' * 60}")
    print(f"Documents evaluated:          {summary['documents']}")
    print(f"Document-type accuracy:       {summary['document_type_accuracy']:.0%}")
    print(f"Field accuracy (exact match): {summary['field_accuracy']:.0%} "
          f"({summary['correct_fields']}/{summary['total_fields']})")
    print(f"Wrong fields caught by the confidence flag: {summary['wrong_fields_that_were_flagged']}")
    print(f"Right fields left unflagged:                {summary['right_fields_that_were_not_flagged']}")
    print("\nPer-field accuracy:")
    for label, ratio in summary["per_field_accuracy"].items():
        print(f"  {label:<24} {ratio}")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "failures": failures}, f, indent=2)
        print(f"\nFull report written to {args.report}")


if __name__ == "__main__":
    main()
