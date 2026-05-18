from __future__ import annotations

import argparse
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_INPUT_DIR = REPO_ROOT / "blocking"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR
DEFAULT_SPLITS = ("train", "valid", "test")

AFFILIATION_MARKER = " COL Affiliation VAL "
RESEARCH_FIELD = "COL Research Interests VAL"
PAPERS_MARKER = " COL Papers VAL "
FIELDS = ("Name", "Affiliation", "Research Interests", "Papers", "Projects")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add an empty Research Interests field to the right entity in "
            "HierGAT train/valid/test txt files."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names without .txt suffix.",
    )
    return parser.parse_args()


def ensure_right_research(right: str, line_no: int, path: Path) -> tuple[str, bool]:
    if f" {RESEARCH_FIELD} " in f" {right} ":
        return right, False

    if AFFILIATION_MARKER not in right:
        raise ValueError(f"{path}:{line_no} right entity is missing Affiliation")
    if PAPERS_MARKER not in right:
        raise ValueError(f"{path}:{line_no} right entity is missing Papers")

    return right.replace(PAPERS_MARKER, f" {RESEARCH_FIELD} {PAPERS_MARKER}", 1), True


def normalize_empty_values(entity: str, line_no: int, path: Path) -> str:
    markers = [f"COL {field} VAL" for field in FIELDS]
    positions = []
    for marker in markers:
        position = entity.find(marker)
        if position == -1:
            raise ValueError(f"{path}:{line_no} entity is missing {marker}")
        positions.append(position)

    if positions != sorted(positions):
        raise ValueError(f"{path}:{line_no} entity fields are not in expected order")

    values = []
    for index, marker in enumerate(markers):
        value_start = positions[index] + len(marker)
        value_end = positions[index + 1] if index + 1 < len(positions) else len(entity)
        value = entity[value_start:value_end].strip()
        values.append(value if value else "[NULL]")

    return " ".join(f"{marker} {value}" for marker, value in zip(markers, values))


def convert_line(line: str, line_no: int, path: Path) -> tuple[str, bool]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        raise ValueError(f"{path}:{line_no} expected 3 tab-separated columns, got {len(parts)}")

    left, right, label = parts
    right, _ = ensure_right_research(right, line_no, path)
    left = normalize_empty_values(left, line_no, path)
    right = normalize_empty_values(right, line_no, path)
    new_line = f"{left}\t{right}\t{label}\n"
    return new_line, new_line != line


def convert_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    total = 0
    changed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            total += 1
            new_line, did_change = convert_line(line, line_no, input_path)
            changed += int(did_change)
            dst.write(new_line)

    return total, changed


def main() -> None:
    args = parse_args()
    for split in args.splits:
        input_path = args.input_dir / f"{split}.txt"
        output_path = args.output_dir / f"{split}.txt"
        total, changed = convert_file(input_path, output_path)
        print(f"{split}: wrote {output_path} ({total} rows, changed {changed})")


if __name__ == "__main__":
    main()
