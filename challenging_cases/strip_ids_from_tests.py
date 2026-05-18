from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILES = [
    PROJECT_ROOT / "challenging_cases" / "affiliation_shifts" / "test.txt",
    PROJECT_ROOT / "challenging_cases" / "missing_attributes" / "test.txt",
    PROJECT_ROOT / "challenging_cases" / "name_ambiguity" / "test.txt",
]


def strip_ids(input_path: Path) -> Path:
    output_path = input_path.with_name("test_no_ids.txt")
    rows = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as fin:
        for line_number, line in enumerate(fin, start=1):
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(
                    f"{input_path} line {line_number}: expected 5 columns, got {len(parts)}"
                )
            _id_left, _id_right, record_left, record_right, label = parts
            rows.append("\t".join([record_left, record_right, label]))

    with output_path.open("w", encoding="utf-8-sig", newline="") as fout:
        fout.write("\n".join(rows))
        if rows:
            fout.write("\n")

    return output_path


def main() -> None:
    for input_path in FILES:
        output_path = strip_ids(input_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
