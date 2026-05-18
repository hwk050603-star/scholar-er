from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_CASES = {
    "affiliation_shifts": REPO_ROOT / "challenging_cases" / "affiliation_shifts" / "test.txt",
    "missing_attributes": REPO_ROOT / "challenging_cases" / "missing_attributes" / "test.txt",
    "name_ambiguity": REPO_ROOT / "challenging_cases" / "name_ambiguity" / "test.txt",
}
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "challenging_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert challenging case test.txt files to ComEM CSV format."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--affiliation-shifts", type=Path, default=DEFAULT_CASES["affiliation_shifts"])
    parser.add_argument("--missing-attributes", type=Path, default=DEFAULT_CASES["missing_attributes"])
    parser.add_argument("--name-ambiguity", type=Path, default=DEFAULT_CASES["name_ambiguity"])
    return parser.parse_args()


def read_challenge_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["id_left", "id_right", "record_left", "record_right", "label"],
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")
    df["label"] = df["label"].astype(str).str.strip()

    bad_labels = sorted(set(df["label"]) - {"0", "1"})
    if bad_labels:
        raise ValueError(f"{path}: unexpected labels: {bad_labels}")
    return df


def split_values(text: object) -> list[str]:
    normalized = str(text or "").strip()
    if normalized.startswith("COL "):
        normalized = normalized[4:]

    values: list[str] = []
    for segment in normalized.split(" COL "):
        if " VAL " not in segment:
            continue
        _field, value = segment.split(" VAL ", 1)
        values.append(value.strip())
    return values


def to_comem_left(text: object) -> str:
    labels = ["name", "affiliation", "research interests", "papers", "projects"]
    values = (split_values(text) + [""] * 5)[:5]
    return ", ".join(f"{label}: {value}" for label, value in zip(labels, values))


def to_comem_right(text: object) -> str:
    values = (split_values(text) + [""] * 4)[:4]
    labels = ["name", "affiliation", "research interests", "papers", "projects"]
    full_values = [values[0], values[1], "", values[2], values[3]]
    return ", ".join(f"{label}: {value}" for label, value in zip(labels, full_values))


def convert_case(name: str, input_path: Path, output_dir: Path) -> Path:
    df = read_challenge_test(input_path)
    df["record_left"] = df["record_left"].apply(to_comem_left)
    df["record_right"] = df["record_right"].apply(to_comem_right)
    df["label"] = df["label"].map({"1": "True", "0": "False"})

    output = df[["id_left", "id_right", "record_left", "record_right", "label"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"{name}: input_rows={len(df)}")
    print(f"{name}: output_rows={len(output)}")
    print(f"{name}: positive_rows={int((output['label'] == 'True').sum())}")
    print(f"{name}: negative_rows={int((output['label'] == 'False').sum())}")
    print(f"{name}: output={output_path.resolve()}")
    return output_path


def main() -> None:
    args = parse_args()
    cases = {
        "affiliation_shifts": args.affiliation_shifts,
        "missing_attributes": args.missing_attributes,
        "name_ambiguity": args.name_ambiguity,
    }
    for name, input_path in cases.items():
        convert_case(name, input_path, args.output_dir)


if __name__ == "__main__":
    main()
