from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

from datasets import Dataset, load_dataset


DATASET_NAME = "isaacus/open-australian-legal-qa"
SPLIT_NAME = "train"

RANDOM_SEED = 42
NUM_DUPLICATE_ROW_SAMPLES = 10
NUM_NORMAL_ROW_SAMPLES = 10

OUTPUT_DIR = Path("data/inspection")
REPORT_PATH = OUTPUT_DIR / "inspection_report.md"
DUPLICATE_VERSION_GROUPS_PATH = (
    OUTPUT_DIR / "duplicate_version_id_groups.jsonl"
)
DUPLICATE_PASSAGE_GROUPS_PATH = (
    OUTPUT_DIR / "duplicate_passage_groups.jsonl"
)
DUPLICATE_ROW_SAMPLES_PATH = (
    OUTPUT_DIR / "duplicate_row_samples.jsonl"
)
NORMAL_ROW_SAMPLES_PATH = (
    OUTPUT_DIR / "normal_row_samples.jsonl"
)


def is_missing(value: Any) -> bool:
    """Return True for null, empty, or whitespace-only values."""
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    return False


def normalize_text(value: Any) -> str:
    """
    Normalize text for exact-content comparison.

    This is still exact duplicate detection after normalization:
    - strip leading/trailing whitespace;
    - lowercase;
    - collapse repeated whitespace.

    It does not use embeddings or semantic similarity.
    """
    if not isinstance(value, str):
        return ""

    text = value.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def get_source(row: dict[str, Any]) -> dict[str, Any]:
    """Return the nested source dictionary, or an empty dictionary."""
    source = row.get("source")
    return source if isinstance(source, dict) else {}


def row_summary(
    index: int,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Create a compact representation of one dataset row."""
    source = get_source(row)

    return {
        "dataset_index": index,
        "version_id": source.get("version_id"),
        "citation": source.get("citation"),
        "document_type": source.get("type"),
        "jurisdiction": source.get("jurisdiction"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "passage": source.get("text"),
    }


def print_row(
    index: int,
    row: dict[str, Any],
) -> None:
    """Print one row in a readable format."""
    item = row_summary(index, row)

    print("\n" + "-" * 100)
    print(f"Dataset index: {item['dataset_index']}")
    print(f"Version ID:    {item['version_id']}")
    print(f"Citation:      {item['citation']}")
    print(f"Document type: {item['document_type']}")
    print(f"Jurisdiction:  {item['jurisdiction']}")
    print(f"\nQuestion:\n{item['question']}")
    print(f"\nAnswer:\n{item['answer']}")
    print(f"\nPassage:\n{item['passage']}")


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    """Write dictionaries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_markdown_row(
    lines: list[str],
    index: int,
    row: dict[str, Any],
) -> None:
    """Append one compact dataset row to the Markdown report."""
    item = row_summary(index, row)

    lines.extend(
        [
            f"- Row `{index}`",
            f"  - Version ID: `{item['version_id']}`",
            f"  - Citation: {item['citation']}",
            f"  - Question: {item['question']}",
        ]
    )


def build_groups(
    data: Dataset,
) -> tuple[
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, list[int]],
    dict[tuple[str, str], list[int]],
]:
    """
    Group rows by version ID, normalized question, normalized passage,
    and normalized question-passage pair.
    """
    version_id_groups: dict[str, list[int]] = defaultdict(list)
    question_groups: dict[str, list[int]] = defaultdict(list)
    passage_groups: dict[str, list[int]] = defaultdict(list)
    pair_groups: dict[tuple[str, str], list[int]] = defaultdict(list)

    for index, row in enumerate(data):
        source = get_source(row)

        version_id = normalize_text(source.get("version_id"))
        question = normalize_text(row.get("question"))
        passage = normalize_text(source.get("text"))

        if version_id:
            version_id_groups[version_id].append(index)

        if question:
            question_groups[question].append(index)

        if passage:
            passage_groups[passage].append(index)

        if question and passage:
            pair_groups[(question, passage)].append(index)

    return (
        version_id_groups,
        question_groups,
        passage_groups,
        pair_groups,
    )


def only_repeated(
    groups: dict[Any, list[int]],
) -> dict[Any, list[int]]:
    """Keep only groups containing more than one row."""
    return {
        key: indices
        for key, indices in groups.items()
        if len(indices) > 1
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )

    report_lines: list[str] = [
        "# Dataset Inspection Report",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- Split: `{SPLIT_NAME}`",
        f"- Total rows: **{len(data)}**",
        "",
    ]

    # ------------------------------------------------------------------
    # 1. Dataset structure
    # ------------------------------------------------------------------
    print("Dataset")
    print(data)
    print(f"\nTotal rows: {data.num_rows}")
    print(f"Columns: {data.column_names}")
    print(f"Features:\n{data.features}")

    report_lines.extend(
        [
            "## Dataset Structure",
            "",
            f"- Columns: `{data.column_names}`",
            f"- Features: `{data.features}`",
            "",
        ]
    )

    # ------------------------------------------------------------------
    # 2. Missing fields
    # ------------------------------------------------------------------
    top_level_fields = [
        "question",
        "answer",
        "text",
        "prompt",
        "source",
    ]

    source_fields = [
        "version_id",
        "type",
        "jurisdiction",
        "source",
        "citation",
        "url",
        "text",
    ]

    print("\nMissing top-level fields")
    report_lines.extend(
        [
            "## Missing Fields",
            "",
            "### Top-level fields",
            "",
        ]
    )

    for field in top_level_fields:
        missing_count = sum(
            is_missing(row.get(field))
            for row in data
        )
        missing_percentage = missing_count / len(data) * 100

        print(
            f"  {field:15s}: "
            f"{missing_count:4d} "
            f"({missing_percentage:.2f}%)"
        )

        report_lines.append(
            f"- `{field}`: {missing_count} "
            f"({missing_percentage:.2f}%)"
        )

    print("\nMissing nested source fields")
    report_lines.extend(
        [
            "",
            "### Nested source fields",
            "",
        ]
    )

    for field in source_fields:
        missing_count = 0

        for row in data:
            source = row.get("source")

            if not isinstance(source, dict):
                missing_count += 1
            elif is_missing(source.get(field)):
                missing_count += 1

        missing_percentage = missing_count / len(data) * 100

        print(
            f"  source.{field:15s}: "
            f"{missing_count:4d} "
            f"({missing_percentage:.2f}%)"
        )

        report_lines.append(
            f"- `source.{field}`: {missing_count} "
            f"({missing_percentage:.2f}%)"
        )

    # ------------------------------------------------------------------
    # 3. Build duplicate groups
    # ------------------------------------------------------------------
    (
        version_id_groups,
        question_groups,
        passage_groups,
        pair_groups,
    ) = build_groups(data)

    repeated_version_ids = only_repeated(version_id_groups)
    repeated_questions = only_repeated(question_groups)
    repeated_passages = only_repeated(passage_groups)
    repeated_pairs = only_repeated(pair_groups)

    # ------------------------------------------------------------------
    # 4. Unique and repeated version IDs
    # ------------------------------------------------------------------
    total_non_missing_version_ids = sum(
        len(indices)
        for indices in version_id_groups.values()
    )

    print("\nVersion IDs")
    print(
        f"  Non-missing version IDs: "
        f"{total_non_missing_version_ids}"
    )
    print(
        f"  Unique version IDs:      "
        f"{len(version_id_groups)}"
    )
    print(
        f"  Repeated version IDs:    "
        f"{len(repeated_version_ids)}"
    )

    report_lines.extend(
        [
            "",
            "## Version IDs",
            "",
            (
                f"- Non-missing version IDs: "
                f"**{total_non_missing_version_ids}**"
            ),
            (
                f"- Unique version IDs: "
                f"**{len(version_id_groups)}**"
            ),
            (
                f"- Repeated version IDs: "
                f"**{len(repeated_version_ids)}**"
            ),
            "",
        ]
    )

    duplicate_version_records: list[dict[str, Any]] = []

    if repeated_version_ids:
        print("\nRepeated version_id groups")
        report_lines.extend(
            [
                "### Repeated version ID groups",
                "",
            ]
        )

        for group_number, (version_id, indices) in enumerate(
            sorted(
                repeated_version_ids.items(),
                key=lambda item: (-len(item[1]), item[0]),
            ),
            start=1,
        ):
            print("\n" + "=" * 100)
            print(f"VERSION-ID GROUP {group_number}")
            print(f"Version ID: {version_id}")
            print(f"Occurrences: {len(indices)}")
            print(f"Dataset indices: {indices}")

            report_lines.extend(
                [
                    f"#### Group {group_number}",
                    "",
                    f"- Version ID: `{version_id}`",
                    f"- Occurrences: {len(indices)}",
                    f"- Dataset indices: `{indices}`",
                ]
            )

            rows = []

            for index in indices:
                summary = row_summary(index, data[index])
                rows.append(summary)

                print(
                    f"  - row {index}: "
                    f"{summary['citation']!r}"
                )
                append_markdown_row(
                    report_lines,
                    index,
                    data[index],
                )

            report_lines.append("")

            duplicate_version_records.append(
                {
                    "version_id": version_id,
                    "dataset_indices": indices,
                    "rows": rows,
                }
            )
    else:
        print("  No repeated version IDs found.")
        report_lines.append(
            "- No repeated version IDs were found."
        )

    write_jsonl(
        DUPLICATE_VERSION_GROUPS_PATH,
        duplicate_version_records,
    )

    # ------------------------------------------------------------------
    # 5. Exact duplicate summary
    # ------------------------------------------------------------------
    print("\nExact duplicate summary")
    print(
        f"  Repeated normalized questions: "
        f"{len(repeated_questions)}"
    )
    print(
        f"  Repeated normalized passages:  "
        f"{len(repeated_passages)}"
    )
    print(
        f"  Repeated normalized pairs:     "
        f"{len(repeated_pairs)}"
    )

    report_lines.extend(
        [
            "",
            "## Exact Duplicate Summary",
            "",
            (
                f"- Repeated normalized questions: "
                f"**{len(repeated_questions)}**"
            ),
            (
                f"- Repeated normalized passages: "
                f"**{len(repeated_passages)}**"
            ),
            (
                f"- Repeated normalized "
                f"question-passage pairs: "
                f"**{len(repeated_pairs)}**"
            ),
            "",
        ]
    )

    # ------------------------------------------------------------------
    # 6. Passage duplicates across different version IDs
    # ------------------------------------------------------------------
    cross_version_duplicate_passages: dict[
        str,
        list[int],
    ] = {}

    for normalized_passage, indices in repeated_passages.items():
        version_ids = {
            normalize_text(
                get_source(data[index]).get("version_id")
            )
            for index in indices
        }
        version_ids.discard("")

        if len(version_ids) > 1:
            cross_version_duplicate_passages[
                normalized_passage
            ] = indices

    print("\nDuplicate passages across different version IDs")
    print(
        f"  Groups: "
        f"{len(cross_version_duplicate_passages)}"
    )

    report_lines.extend(
        [
            "## Duplicate Passages Across Different Version IDs",
            "",
            (
                f"- Number of groups: "
                f"**{len(cross_version_duplicate_passages)}**"
            ),
            "",
        ]
    )

    duplicate_passage_records: list[dict[str, Any]] = []

    for group_number, (
        normalized_passage,
        indices,
    ) in enumerate(
        sorted(
            cross_version_duplicate_passages.items(),
            key=lambda item: (-len(item[1]), item[0]),
        ),
        start=1,
    ):
        version_ids = sorted(
            {
                get_source(data[index]).get("version_id")
                for index in indices
            }
        )

        print("\n" + "=" * 100)
        print(f"CROSS-VERSION PASSAGE GROUP {group_number}")
        print(f"Occurrences: {len(indices)}")
        print(f"Dataset indices: {indices}")
        print(f"Version IDs: {version_ids}")
        print(
            "Passage preview: "
            f"{normalized_passage[:300]!r}"
        )

        report_lines.extend(
            [
                f"### Group {group_number}",
                "",
                f"- Occurrences: {len(indices)}",
                f"- Dataset indices: `{indices}`",
                f"- Version IDs: `{version_ids}`",
                (
                    f"- Passage preview: "
                    f"`{normalized_passage[:300]}`"
                ),
            ]
        )

        rows = []

        for index in indices:
            summary = row_summary(index, data[index])
            rows.append(summary)
            append_markdown_row(
                report_lines,
                index,
                data[index],
            )

        report_lines.append("")

        duplicate_passage_records.append(
            {
                "normalized_passage": normalized_passage,
                "dataset_indices": indices,
                "version_ids": version_ids,
                "rows": rows,
            }
        )

    if not cross_version_duplicate_passages:
        report_lines.append(
            "- No normalized passage text was repeated "
            "across different version IDs."
        )
        report_lines.append("")

    write_jsonl(
        DUPLICATE_PASSAGE_GROUPS_PATH,
        duplicate_passage_records,
    )

    # ------------------------------------------------------------------
    # 7. Sample rows involved in any duplication
    # ------------------------------------------------------------------
    duplicate_row_indices: set[int] = set()

    for groups in (
        repeated_version_ids,
        repeated_questions,
        repeated_passages,
        repeated_pairs,
    ):
        for indices in groups.values():
            duplicate_row_indices.update(indices)

    sorted_duplicate_indices = sorted(duplicate_row_indices)
    duplicate_sample_indices = sorted_duplicate_indices[
        :NUM_DUPLICATE_ROW_SAMPLES
    ]

    print(
        f"\nDuplicate-row sample "
        f"({len(duplicate_sample_indices)} of "
        f"{len(sorted_duplicate_indices)} "
        f"duplicate-involved rows)"
    )

    duplicate_sample_rows = []

    for index in duplicate_sample_indices:
        print_row(index, data[index])
        duplicate_sample_rows.append(
            row_summary(index, data[index])
        )

    write_jsonl(
        DUPLICATE_ROW_SAMPLES_PATH,
        duplicate_sample_rows,
    )

    report_lines.extend(
        [
            "## Duplicate-row Sample",
            "",
            (
                f"- Sampled {len(duplicate_sample_indices)} "
                f"of {len(sorted_duplicate_indices)} rows "
                f"involved in a duplicate or repeated "
                f"version-ID group."
            ),
            "",
        ]
    )

    for index in duplicate_sample_indices:
        append_markdown_row(
            report_lines,
            index,
            data[index],
        )

    # ------------------------------------------------------------------
    # 8. Sample 10 ordinary rows
    # ------------------------------------------------------------------
    normal_candidate_indices = [
        index
        for index in range(len(data))
        if index not in duplicate_row_indices
    ]

    rng = random.Random(RANDOM_SEED)
    normal_sample_size = min(
        NUM_NORMAL_ROW_SAMPLES,
        len(normal_candidate_indices),
    )
    normal_sample_indices = rng.sample(
        normal_candidate_indices,
        normal_sample_size,
    )

    print(
        f"\nNormal-row sample "
        f"({len(normal_sample_indices)} rows)"
    )

    normal_sample_rows = []

    for index in normal_sample_indices:
        print_row(index, data[index])
        normal_sample_rows.append(
            row_summary(index, data[index])
        )

    write_jsonl(
        NORMAL_ROW_SAMPLES_PATH,
        normal_sample_rows,
    )

    report_lines.extend(
        [
            "",
            "## Normal-row Sample",
            "",
            (
                f"- Random seed: `{RANDOM_SEED}`"
            ),
            (
                f"- Sample size: "
                f"{len(normal_sample_indices)}"
            ),
            "",
        ]
    )

    for index in normal_sample_indices:
        append_markdown_row(
            report_lines,
            index,
            data[index],
        )

    # ------------------------------------------------------------------
    # 9. Save Markdown report
    # ------------------------------------------------------------------
    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\nSaved inspection outputs")
    print(f"  Markdown report: {REPORT_PATH.resolve()}")
    print(
        "  Duplicate version groups: "
        f"{DUPLICATE_VERSION_GROUPS_PATH.resolve()}"
    )
    print(
        "  Cross-version duplicate passages: "
        f"{DUPLICATE_PASSAGE_GROUPS_PATH.resolve()}"
    )
    print(
        "  Duplicate row samples: "
        f"{DUPLICATE_ROW_SAMPLES_PATH.resolve()}"
    )
    print(
        "  Normal row samples: "
        f"{NORMAL_ROW_SAMPLES_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()