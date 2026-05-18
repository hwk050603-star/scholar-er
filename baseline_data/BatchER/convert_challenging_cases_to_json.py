from __future__ import annotations

import argparse
from pathlib import Path

from convert_blocking_txt_to_json import process_txt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_CASES = {
    "missing_attributes": REPO_ROOT / "challenging_cases" / "missing_attributes" / "test_no_ids.txt",
    "affiliation_shifts": REPO_ROOT / "challenging_cases" / "affiliation_shifts" / "test_no_ids.txt",
    "name_ambiguity": REPO_ROOT / "challenging_cases" / "name_ambiguity" / "test_no_ids.txt",
}
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "challenging_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert challenging case test_no_ids.txt files to BatchER JSON format."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--missing-attributes", type=Path, default=DEFAULT_CASES["missing_attributes"])
    parser.add_argument("--affiliation-shifts", type=Path, default=DEFAULT_CASES["affiliation_shifts"])
    parser.add_argument("--name-ambiguity", type=Path, default=DEFAULT_CASES["name_ambiguity"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = {
        "missing_attributes": args.missing_attributes,
        "affiliation_shifts": args.affiliation_shifts,
        "name_ambiguity": args.name_ambiguity,
    }

    for name, input_path in cases.items():
        output_path = args.output_dir / f"{name}.json"
        process_txt(input_path, output_path)


if __name__ == "__main__":
    main()
