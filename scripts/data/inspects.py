#!/usr/bin/env python3
"""Inspect version IDs in the raw Open Australian Legal QA dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT = Path(
    "data/raw/embed/open-australian-legal-qa-v2.0.0/qa.jsonl"
)


def inspect_version_ids(input_path: Path) -> dict:
    """Count version IDs and retain row indices for repeated IDs."""
    version_rows: dict[str, list[int]] = defaultdict(list)
    total_rows = 0
    missing_rows: list[int] = []

    with input_path.open(encoding="utf-8-sig") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue

            total_rows += 1
            row = json.loads(line)
            source = row.get("source")
            source = source if isinstance(source, dict) else {}
            version_id = source.get("version_id")

            if not isinstance(version_id, str) or not version_id.strip():
                missing_rows.append(row_index)
                continue

            # Casefolding prevents capitalization-only differences from being
            # counted as distinct source documents.
            normalized_id = version_id.strip().casefold()
            version_rows[normalized_id].append(row_index)

    repeated = {
        version_id: indices
        for version_id, indices in sorted(version_rows.items())
        if len(indices) > 1
    }

    return {
        "input": str(input_path),
        "total_rows": total_rows,
        "non_missing_version_ids": total_rows - len(missing_rows),
        "missing_version_ids": len(missing_rows),
        "missing_row_indices": missing_rows,
        "unique_version_ids": len(version_rows),
        "repeated_version_id_groups": len(repeated),
        "extra_rows_from_repeated_ids": sum(
            len(indices) - 1 for indices in repeated.values()
        ),
        "repeated_version_ids": repeated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete result as JSON.",
    )
    args = parser.parse_args()

    report = inspect_version_ids(args.input)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"Input:                       {report['input']}")
    print(f"Total rows:                  {report['total_rows']}")
    print(f"Non-missing version IDs:     {report['non_missing_version_ids']}")
    print(f"Missing version IDs:         {report['missing_version_ids']}")
    print(f"Unique version IDs:          {report['unique_version_ids']}")
    print(
        "Repeated version-ID groups: "
        f"{report['repeated_version_id_groups']}"
    )
    print(
        "Extra rows from repeats:     "
        f"{report['extra_rows_from_repeated_ids']}"
    )

    if report["repeated_version_ids"]:
        print("\nRepeated version IDs:")
        for version_id, indices in report["repeated_version_ids"].items():
            print(f"- {version_id}: {len(indices)} rows at indices {indices}")


if __name__ == "__main__":
    main()
