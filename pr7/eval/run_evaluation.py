"""Run PR7 invoice extraction against expected.json and save auditable results.

Run from the pr7 directory only when willing to spend model quota:
    python eval/run_evaluation.py --run-model --include-own
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app import llm
from app.extraction import process


FIELD_ALIASES = {
    "document_type": "doc_type",
    "valid_until": "due_date",
    "total_without_vat": "total_no_vat",
    "total": "total_to_pay",
    "amount": "sum",
}
EXPECTED_RULES = {
    "arithmetic_line": {"line_math"},
    "buyer": {"buyer_reference"},
    "iban_checksum": {"iban_checksum"},
    "iban_registry": {"supplier_reference"},
    "missing_required": {"required"},
    "document_type": set(),
}
CRITICAL_FIELDS = {"supplier.iban", "supplier.code", "total_to_pay"}


def _mapped_path(path: str) -> str:
    parts = path.split(".")
    parts = [FIELD_ALIASES.get(part, part) for part in parts]
    return ".".join(parts)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = _mapped_path(f"{prefix}.{key}" if prefix else key)
            result.update(_flatten(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{prefix}.{index}"))
        if not value and prefix:
            result[prefix] = []
    elif prefix:
        result[prefix] = value
    return result


def _normalized(path: str, value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if path == "doc_type":
        normalized_type = text.casefold()
        if "рахунок" in normalized_type:
            return "invoice"
        if "накладн" in normalized_type:
            return "deliverynote"
    if path.endswith(".iban"):
        return re.sub(r"\s+", "", text).upper()
    if path.endswith(".code"):
        return re.sub(r"\s+", "", text)
    if path.endswith((".quantity", ".price", ".sum", "total_no_vat", "vat", "total_to_pay")):
        candidate = re.sub(r"[\s\u00a0]", "", text).replace(",", ".")
        try:
            return Decimal(candidate).normalize()
        except InvalidOperation:
            return text
    if path in {"date", "due_date"}:
        for date_format in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(text, date_format).date().isoformat()
            except ValueError:
                pass
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("ʼ", "'").replace("’", "'").replace("`", "'")
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _load_cases(include_own: bool) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    expected_path = PROJECT_DIR / "samples" / "expected.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    cases = []
    for relative_path, record in expected.items():
        path = PROJECT_DIR / "samples" / relative_path
        if path.is_file():
            cases.append((relative_path, record, {"expected_source": relative_path}))

    if include_own:
        manifest_path = PROJECT_DIR / "samples" / "own" / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(
                "Missing samples/own/manifest.json. Add each own sample and its "
                "expected_source from samples/expected.json."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            relative_path = f"own/{entry['file']}"
            expected_source = entry["expected_source"]
            if expected_source not in expected:
                raise SystemExit(
                    f"Unknown expected_source {expected_source!r} in own manifest."
                )
            path = PROJECT_DIR / "samples" / relative_path
            if not path.is_file():
                raise SystemExit(f"Own sample image does not exist: {path}")
            cases.append(
                (
                    relative_path,
                    expected[expected_source],
                    {"expected_source": expected_source},
                )
            )
    return cases


def _expected_invoice(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_type": fields.get("document_type"),
        "number": fields.get("number"),
        "date": fields.get("date"),
        "valid_until": fields.get("valid_until"),
        "supplier": fields.get("supplier"),
        "buyer": fields.get("buyer"),
        "items": fields.get("items"),
        "total_without_vat": fields.get("total_without_vat"),
        "vat": fields.get("vat"),
        "total": fields.get("total"),
    }


def _evaluate_fields(expected: dict[str, Any], actual: dict[str, Any]) -> list[dict[str, Any]]:
    expected_flat = _flatten(expected)
    actual_flat = _flatten(actual)
    paths = sorted(set(expected_flat) | set(actual_flat))
    comparisons = []
    for path in paths:
        expected_value = expected_flat.get(path)
        actual_value = actual_flat.get(path)
        if _normalized(path, expected_value) == _normalized(path, actual_value):
            label = "correct"
        elif expected_value is None and actual_value is not None:
            label = "hallucination"
        elif expected_value is not None and actual_value is None:
            label = "omission"
        else:
            label = "error"
        comparisons.append(
            {
                "field": path,
                "expected": expected_value,
                "actual": actual_value,
                "label": label,
            }
        )
    return comparisons


def _issue_caught(field: str, issues: list[dict[str, Any]]) -> bool:
    return any(
        issue.get("field") == field
        or field.startswith(f"{issue.get('field', '')}.")
        or issue.get("field", "").startswith(f"{field}.")
        for issue in issues
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PR7 extraction samples.")
    parser.add_argument(
        "--run-model",
        action="store_true",
        help="Send every selected image to the configured model; this uses API quota.",
    )
    parser.add_argument(
        "--include-own",
        action="store_true",
        help="Include files listed in samples/own/manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: eval/results-<timestamp>.json).",
    )
    args = parser.parse_args()

    if not args.run_model:
        raise SystemExit(
            "No model was called. To run real multimodal evaluation explicitly, "
            "use --run-model; this sends one API request per image."
        )
    llm.MOCK_LLM = False

    cases = _load_cases(args.include_own)
    records = []
    for relative_path, expected_record, metadata in cases:
        image_path = PROJECT_DIR / "samples" / relative_path
        try:
            result = process(image_path.read_bytes())
            actual_document = result.document or {}
            issues = result.issues
            comparisons = _evaluate_fields(
                _expected_invoice(expected_record["fields"]),
                actual_document,
            )
            for comparison in comparisons:
                comparison["caught_by_rule"] = (
                    comparison["label"] not in {"correct"}
                    and _issue_caught(comparison["field"], issues)
                )
            actual_rules = {issue.get("rule") for issue in issues}
            expected_checks = set(expected_record.get("expected_checks", []))
            matched_checks = []
            missing_checks = []
            for check in expected_checks:
                mapped_rules = EXPECTED_RULES.get(check, set())
                matched = (
                    result.decision == "reject"
                    if check == "document_type"
                    else bool(mapped_rules & actual_rules)
                )
                (matched_checks if matched else missing_checks).append(check)
            records.append(
                {
                    "file": relative_path,
                    "variant": expected_record.get("variant", "own"),
                    "expected_source": metadata["expected_source"],
                    "expected_decision": expected_record.get("expected_decision"),
                    "actual_decision": result.decision,
                    "decision_matches": (
                        result.decision == expected_record.get("expected_decision")
                    ),
                    "expected_checks": sorted(expected_checks),
                    "matched_expected_checks": sorted(matched_checks),
                    "missing_expected_checks": sorted(missing_checks),
                    "unexpected_rules": sorted(
                        rule
                        for rule in actual_rules
                        if rule
                        and not any(
                            rule in EXPECTED_RULES.get(check, set())
                            for check in expected_checks
                        )
                    ),
                    "issues": issues,
                    "field_comparisons": comparisons,
                    "tokens": result.usage,
                    "elapsed": result.elapsed,
                    "document": actual_document,
                    "raw_response": (result.usage or {}).get("raw_response"),
                    "silent_field_errors": [
                        comparison
                        for comparison in comparisons
                        if comparison["label"] in {"error", "hallucination"}
                        and not comparison["caught_by_rule"]
                    ],
                }
            )
        except Exception as exc:
            records.append(
                {
                    "file": relative_path,
                    "variant": expected_record.get("variant", "own"),
                    "expected_source": metadata["expected_source"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    output_path = args.output or (
        PROJECT_DIR
        / "eval"
        / f"results-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model": llm.MODEL_NAME,
                "mock": False,
                "sample_count": len(records),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("variant", "unknown")].append(record)
    print(f"Saved {len(records)} results to {output_path}")
    for variant, group in sorted(grouped.items()):
        labels = [
            item["label"]
            for record in group
            for item in record.get("field_comparisons", [])
        ]
        print(
            f"{variant}: files={len(group)}, "
            f"correct={labels.count('correct')}, error={labels.count('error')}, "
            f"omission={labels.count('omission')}, "
            f"hallucination={labels.count('hallucination')}, "
            f"caught={sum(bool(item.get('caught_by_rule')) for record in group for item in record.get('field_comparisons', []))}, "
            f"silent={sum(len(record.get('silent_field_errors', [])) for record in group)}, "
            f"decision_matches={sum(bool(record.get('decision_matches')) for record in group)}/{len(group)}, "
            f"prompt_tokens={sum((record.get('tokens') or {}).get('prompt_tokens', 0) for record in group)}, "
            f"run_errors={sum('error' in record for record in group)}"
        )
    critical = [
        comparison
        for record in records
        for comparison in record.get("field_comparisons", [])
        if comparison["field"] in CRITICAL_FIELDS
    ]
    print(
        "critical fields: "
        f"correct={sum(item['label'] == 'correct' for item in critical)}/"
        f"{len(critical)}"
    )
    return 0 if all("error" not in record for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
