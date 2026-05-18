from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

FIELDS = ("Name", "Affiliation", "Research Interests", "Papers", "Projects")
DEFAULT_CASES = {
    "affiliation_shifts": REPO_ROOT / "challenging_cases" / "affiliation_shifts" / "test_no_ids.txt",
    "missing_attributes": REPO_ROOT / "challenging_cases" / "missing_attributes" / "test_no_ids.txt",
    "name_ambiguity": REPO_ROOT / "challenging_cases" / "name_ambiguity" / "test_no_ids.txt",
}
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "challenging_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert challenging case test_no_ids.txt files for HierGAT by "
            "ensuring both left and right records contain all expected fields."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--affiliation-shifts", type=Path, default=DEFAULT_CASES["affiliation_shifts"])
    parser.add_argument("--missing-attributes", type=Path, default=DEFAULT_CASES["missing_attributes"])
    parser.add_argument("--name-ambiguity", type=Path, default=DEFAULT_CASES["name_ambiguity"])
    return parser.parse_args()


def extract_fields(entity: str) -> dict[str, str]:
    text = str(entity or "").strip()
    markers = [(field, f"COL {field} VAL") for field in FIELDS]
    positions: list[tuple[int, str, str]] = []

    for field, marker in markers:
        pos = text.find(marker)
        if pos != -1:
            positions.append((pos, field, marker))

    positions.sort(key=lambda item: item[0])
    values = {field: "" for field in FIELDS}

    for index, (pos, field, marker) in enumerate(positions):
        value_start = pos + len(marker)
        value_end = positions[index + 1][0] if index + 1 < len(positions) else len(text)
        values[field] = text[value_start:value_end].strip()

    return values


def normalize_entity(entity: str) -> str:
    values = extract_fields(entity)
    return " ".join(f"COL {field} VAL {values[field]}" for field in FIELDS)


def convert_line(line: str, line_no: int, path: Path) -> str:
    parts = line.rstrip("\n\r").split("\t")
    if len(parts) != 3:
        raise ValueError(f"{path}:{line_no} expected 3 tab-separated columns, got {len(parts)}")

    left, right, label = parts
    if label not in {"0", "1"}:
        raise ValueError(f"{path}:{line_no} expected label 0/1, got {label!r}")

    return f"{normalize_entity(left)}\t{normalize_entity(right)}\t{label}\n"


def convert_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    total = 0
    changed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8-sig", newline="") as src, output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            total += 1
            new_line = convert_line(line, line_no, input_path)
            changed += int(new_line != line)
            dst.write(new_line)

    return total, changed


def main() -> None:
    args = parse_args()
    cases = {
        "affiliation_shifts": args.affiliation_shifts,
        "missing_attributes": args.missing_attributes,
        "name_ambiguity": args.name_ambiguity,
    }

    for name, input_path in cases.items():
        output_path = args.output_dir / f"{name}.txt"
        total, changed = convert_file(input_path, output_path)
        print(f"{name}: wrote {output_path.resolve()} ({total} rows, changed {changed})")


if __name__ == "__main__":
    main()
