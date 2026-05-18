import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "blocking"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert blocking train/valid/test txt files to batcher JSON format."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing train.txt/valid.txt/test.txt. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--blocking-dir",
        type=Path,
        default=None,
        help="Deprecated alias for --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write train.json/valid.json/test.json. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def parse_one_person(block: str) -> str:
    result = {
        "Name": "",
        "Affiliation": "",
        "Research Interests": "",
        "Papers": "",
        "Projects": "",
    }

    text = block.strip()
    if text.startswith("COL "):
        text = text[4:]

    for segment in text.split(" COL "):
        if " VAL " not in segment:
            continue
        field_name, value = segment.split(" VAL ", 1)
        field_name = field_name.strip()
        if field_name in result:
            result[field_name] = value.strip()

    return json.dumps(result, ensure_ascii=False, separators=(", ", ": "))


def process_txt(input_path: Path, output_path: Path) -> None:
    final_result: list[list[str]] = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for line_number, parts in enumerate(reader, start=1):
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(
                    f"{input_path.name} line {line_number}: expected 3 TSV columns, got {len(parts)}"
                )

            left_record, right_record, label = parts
            label = label.strip()
            if label not in {"0", "1"}:
                raise ValueError(f"{input_path.name} line {line_number}: invalid label")

            final_result.append(
                [parse_one_person(left_record), parse_one_person(right_record), label]
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig") as file:
        json.dump(final_result, file, ensure_ascii=False, indent=2)

    print(f"input={input_path}")
    print(f"rows={len(final_result)}")
    print(f"output={output_path}")


def main() -> None:
    args = parse_args()
    input_dir = args.blocking_dir if args.blocking_dir is not None else args.input_dir

    for stem in ("train", "valid", "test"):
        input_path = input_dir / f"{stem}.txt"
        output_path = args.output_dir / f"{stem}.json"
        process_txt(input_path, output_path)


if __name__ == "__main__":
    main()
