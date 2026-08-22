#!/usr/bin/env python3
"""Validate the public BrailleBench data tree and optional checksum manifest."""

import argparse
import hashlib
import json
from pathlib import Path


DATASETS = {
    "gsm8k": 1319,
    "aime24": 30,
    "commonsenseqa": 1221,
    "hotpotqa": 7405,
    "2wikimultihopqa": 12576,
}
GRADES = ("grade1", "grade2")
FORMATS = ("ascii", "unicode", "dots")
PARALLEL_FIELDS = ("id", "question", "answer", "golden_answers", "choices")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            rows.append(row)
    return rows


def get_gold(row: dict):
    if "golden_answers" in row:
        return row["golden_answers"]
    return row.get("answer")


def validate_row(path: Path, index: int, row: dict, dataset: str) -> None:
    if not str(row.get("question", "")).strip():
        raise ValueError(f"{path}:{index + 1}: missing English question")
    if not str(row.get("braille_question", "")).strip():
        raise ValueError(f"{path}:{index + 1}: missing Braille question")
    gold = get_gold(row)
    if gold in (None, "", []):
        raise ValueError(f"{path}:{index + 1}: missing gold answer")

    if dataset in ("gsm8k", "aime24"):
        if not str(row.get("braille_answer", "")).strip():
            raise ValueError(f"{path}:{index + 1}: missing Braille math answer")
    else:
        braille_gold = row.get("braille_golden_answers")
        if not isinstance(braille_gold, list) or not braille_gold:
            raise ValueError(f"{path}:{index + 1}: missing Braille QA gold")

    if dataset == "commonsenseqa":
        choices = row.get("choices")
        braille_choices = row.get("braille_choices")
        if not isinstance(choices, list) or len(choices) != 5:
            raise ValueError(f"{path}:{index + 1}: expected five English choices")
        if not isinstance(braille_choices, list) or len(braille_choices) != 5:
            raise ValueError(f"{path}:{index + 1}: expected five Braille choices")


def validate_data(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    manifest = {
        "schema_version": 1,
        "logical_records": sum(DATASETS.values()),
        "variant_rows": 0,
        "datasets": {},
    }

    for dataset, expected_count in DATASETS.items():
        variants = {}
        baseline = None
        for grade in GRADES:
            for braille_format in FORMATS:
                relative = Path(dataset) / grade / f"{braille_format}.jsonl"
                path = data_dir / relative
                if not path.is_file():
                    raise FileNotFoundError(f"missing benchmark file: {path}")
                rows = read_jsonl(path)
                if len(rows) != expected_count:
                    raise ValueError(
                        f"{path}: expected {expected_count} rows, found {len(rows)}"
                    )
                for index, row in enumerate(rows):
                    validate_row(path, index, row, dataset)

                if baseline is None:
                    baseline = rows
                else:
                    for index, (reference, row) in enumerate(zip(baseline, rows)):
                        for field in PARALLEL_FIELDS:
                            if reference.get(field) != row.get(field):
                                raise ValueError(
                                    f"{path}:{index + 1}: parallel field {field!r} "
                                    "does not match grade1/ascii"
                                )

                variants[f"{grade}/{braille_format}"] = {
                    "path": relative.as_posix(),
                    "records": len(rows),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                manifest["variant_rows"] += len(rows)

        manifest["datasets"][dataset] = {
            "records": expected_count,
            "variants": variants,
        }

    expected_files = len(DATASETS) * len(GRADES) * len(FORMATS)
    actual_files = list(data_dir.glob("*/*/*.jsonl"))
    if len(actual_files) != expected_files:
        raise ValueError(
            f"{data_dir}: expected {expected_files} JSONL files, found {len(actual_files)}"
        )
    return manifest


def check_manifest(actual: dict, manifest_path: Path) -> None:
    expected = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if expected != actual:
        raise ValueError(f"checksum manifest does not match data: {manifest_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "braille")
    parser.add_argument("--check-manifest", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()

    manifest = validate_data(args.data_dir)
    if args.check_manifest:
        check_manifest(manifest, args.check_manifest)
    if args.write_manifest:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        "BrailleBench data OK: "
        f"datasets={len(DATASETS)} files={len(DATASETS) * len(GRADES) * len(FORMATS)} "
        f"logical_records={manifest['logical_records']} "
        f"variant_rows={manifest['variant_rows']}"
    )


if __name__ == "__main__":
    main()
