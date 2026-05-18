import argparse
from pathlib import Path
import re

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_BLOCKING_TEST = REPO_ROOT / "blocking" / "test.txt"
DEFAULT_CANDIDATES = REPO_ROOT / "blocking" / "A_B_blocking_candidates.csv"
DEFAULT_AUG_SAME_NAME = REPO_ROOT / "blocking" / "dataset_aug_same_name.csv"
DEFAULT_DATASET_A = REPO_ROOT / "blocking" / "dataset_A.csv"
DEFAULT_DATASET_B = REPO_ROOT / "blocking" / "dataset_B.csv"
DEFAULT_SAME_NAME = REPO_ROOT / "pretrain" / "dataset" / "pretrain_dataset_same_name.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "A-B.csv"

ID_COL = "Id"
NAME_COL = "Name"
AFFILIATION_COL = "Affiliation"
RESEARCH_COL = "Research Interests"
PAPERS_COL = "Papers"
PROJECTS_COL = "Projects"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the full blocking/test.txt file to A-B.csv."
        )
    )
    parser.add_argument(
        "--blocking-test",
        type=Path,
        default=DEFAULT_BLOCKING_TEST,
        help=f"Path to blocking test.txt. Default: {DEFAULT_BLOCKING_TEST}",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help=f"Path to A_B_blocking_candidates.csv. Default: {DEFAULT_CANDIDATES}",
    )
    parser.add_argument(
        "--aug-same-name",
        type=Path,
        default=DEFAULT_AUG_SAME_NAME,
        help=f"Path to dataset_aug_same_name.csv. Default: {DEFAULT_AUG_SAME_NAME}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write A-B.csv. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--dataset-a", type=Path, default=DEFAULT_DATASET_A)
    parser.add_argument("--dataset-b", type=Path, default=DEFAULT_DATASET_B)
    parser.add_argument("--same-name-csv", type=Path, default=DEFAULT_SAME_NAME)
    return parser.parse_args()


def read_blocking_test(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["record_left", "record_right", "label"],
        dtype=str,
        encoding="utf-8-sig",
    ).fillna("")
    df["label"] = df["label"].astype(str).str.strip()
    return df


def build_id_lookup(path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    candidates["label"] = candidates["label"].astype(str).str.strip()

    key_cols = ["record_left", "record_right", "label"]
    lookup = (
        candidates.sort_values(["id_left", "id_right"], kind="stable")
        .drop_duplicates(subset=key_cols, keep="first")[
            ["id_left", "id_right", *key_cols]
        ]
    )
    return lookup


def add_sample_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_sample_key"] = (
        out["record_left"].astype(str)
        + "\t"
        + out["record_right"].astype(str)
    )
    return out


def build_combined_id_lookup(candidates_path: Path, aug_same_name_path: Path) -> pd.DataFrame:
    frames = []
    for path in [candidates_path, aug_same_name_path]:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
        df["label"] = df["label"].astype(str).str.strip()
        frames.append(df[["id_left", "id_right", "record_left", "record_right", "label"]])
    if not frames:
        raise ValueError("No available ID lookup source files.")

    lookup = pd.concat(frames, ignore_index=True)
    return (
        lookup.sort_values(["id_left", "id_right"], kind="stable")
        .drop_duplicates(["record_left", "record_right", "label"], keep="first")
        .reset_index(drop=True)
    )


def split_multi_value(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def left_first(value: str) -> str:
    parts = split_multi_value(value)
    return parts[0] if parts else ""


def right_tail(value: str) -> str:
    parts = split_multi_value(value)
    if len(parts) >= 3:
        return "|".join(parts[1:3])
    if len(parts) == 2:
        return parts[1]
    if len(parts) == 1:
        return parts[0]
    return ""


def serialize_left_from_row(row: pd.Series) -> str:
    return " ".join(
        [
            f"COL {NAME_COL} VAL {str(row[NAME_COL]).strip()}",
            f"COL {AFFILIATION_COL} VAL {str(row[AFFILIATION_COL]).strip()}",
            f"COL {RESEARCH_COL} VAL {str(row.get(RESEARCH_COL, '')).strip()}",
            f"COL {PAPERS_COL} VAL {left_first(row.get(PAPERS_COL, ''))}",
            f"COL {PROJECTS_COL} VAL {left_first(row.get(PROJECTS_COL, ''))}",
        ]
    ).strip()


def serialize_right_from_row(row: pd.Series) -> str:
    return " ".join(
        [
            f"COL {NAME_COL} VAL {str(row[NAME_COL]).strip()}",
            f"COL {AFFILIATION_COL} VAL {str(row[AFFILIATION_COL]).strip()}",
            f"COL {PAPERS_COL} VAL {right_tail(row.get(PAPERS_COL, ''))}",
            f"COL {PROJECTS_COL} VAL {right_tail(row.get(PROJECTS_COL, ''))}",
        ]
    ).strip()


def serialize_full_left_from_row(row: pd.Series) -> str:
    return " ".join(
        [
            f"COL {NAME_COL} VAL {str(row[NAME_COL]).strip()}",
            f"COL {AFFILIATION_COL} VAL {str(row[AFFILIATION_COL]).strip()}",
            f"COL {RESEARCH_COL} VAL {str(row.get(RESEARCH_COL, '')).strip()}",
            f"COL {PAPERS_COL} VAL {str(row.get(PAPERS_COL, '')).strip()}",
            f"COL {PROJECTS_COL} VAL {str(row.get(PROJECTS_COL, '')).strip()}",
        ]
    ).strip()


def serialize_full_right_from_row(row: pd.Series) -> str:
    return " ".join(
        [
            f"COL {NAME_COL} VAL {str(row[NAME_COL]).strip()}",
            f"COL {AFFILIATION_COL} VAL {str(row[AFFILIATION_COL]).strip()}",
            f"COL {PAPERS_COL} VAL {str(row.get(PAPERS_COL, '')).strip()}",
            f"COL {PROJECTS_COL} VAL {str(row.get(PROJECTS_COL, '')).strip()}",
        ]
    ).strip()


def clean_org(value: object) -> str:
    org = str(value).strip()
    org = re.sub(r"[（(].*?[）)]", "", org)
    org = re.sub(r"[、,，;；]+", " ", org)
    return re.sub(r"\s+", " ", org).strip()


def row_with_clean_org(row: pd.Series) -> pd.Series:
    cleaned = row.copy()
    cleaned[AFFILIATION_COL] = clean_org(row.get(AFFILIATION_COL, ""))
    return cleaned


def add_left_lookup_variants(left_lookup: dict[str, str], row: pd.Series, seq: str) -> None:
    for candidate in (row, row_with_clean_org(row)):
        left_lookup.setdefault(serialize_left_from_row(candidate), seq)
        left_lookup.setdefault(serialize_full_left_from_row(candidate), seq)


def add_right_lookup_variants(right_lookup: dict[str, str], row: pd.Series, seq: str) -> None:
    for candidate in (row, row_with_clean_org(row)):
        right_lookup.setdefault(serialize_right_from_row(candidate), seq)
        right_lookup.setdefault(serialize_full_right_from_row(candidate), seq)


def content_key_from_values(values: list[str], paper_idx: int, project_idx: int) -> tuple[str, str] | None:
    paper = values[paper_idx].strip() if len(values) > paper_idx else ""
    project = values[project_idx].strip() if len(values) > project_idx else ""
    if not paper and not project:
        return None
    return paper, project


def left_content_key(text: str) -> tuple[str, str] | None:
    return content_key_from_values(split_values(text), 3, 4)


def right_content_key(text: str) -> tuple[str, str] | None:
    return content_key_from_values(split_values(text), 2, 3)


def set_unique(mapping: dict[tuple[str, str], str | None], key: tuple[str, str] | None, seq: str) -> None:
    if key is None:
        return
    if key not in mapping:
        mapping[key] = seq


def add_content_lookup_variants(
    left_content_lookup: dict[tuple[str, str], str | None],
    right_content_lookup: dict[tuple[str, str], str | None],
    row: pd.Series,
    seq: str,
) -> None:
    for candidate in (row, row_with_clean_org(row)):
        set_unique(left_content_lookup, left_content_key(serialize_left_from_row(candidate)), seq)
        set_unique(left_content_lookup, left_content_key(serialize_full_left_from_row(candidate)), seq)
        set_unique(right_content_lookup, right_content_key(serialize_right_from_row(candidate)), seq)
        set_unique(right_content_lookup, right_content_key(serialize_full_right_from_row(candidate)), seq)


def build_record_lookup(
    dataset_a: Path,
    dataset_b: Path,
    same_name_csv: Path,
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[tuple[str, str], str | None],
    dict[tuple[str, str], str | None],
]:
    left_lookup: dict[str, str] = {}
    right_lookup: dict[str, str] = {}
    left_content_lookup: dict[tuple[str, str], str | None] = {}
    right_content_lookup: dict[tuple[str, str], str | None] = {}

    record_paths = [
        dataset_a,
        dataset_a.with_name(f"{dataset_a.stem}_aug{dataset_a.suffix}"),
        dataset_b,
        dataset_b.with_name(f"{dataset_b.stem}_aug{dataset_b.suffix}"),
    ]

    for path in record_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
        for _, row in df.iterrows():
            seq = str(row[ID_COL]).strip()
            add_left_lookup_variants(left_lookup, row, seq)
            add_right_lookup_variants(right_lookup, row, seq)
            add_content_lookup_variants(left_content_lookup, right_content_lookup, row, seq)

    same_name_df = pd.read_csv(same_name_csv, dtype=str, encoding="utf-8-sig").fillna("")
    for _, row in same_name_df.iterrows():
        seq = str(row[ID_COL]).strip()
        add_left_lookup_variants(left_lookup, row, seq)
        add_right_lookup_variants(right_lookup, row, seq)
        add_content_lookup_variants(left_content_lookup, right_content_lookup, row, seq)

    return left_lookup, right_lookup, left_content_lookup, right_content_lookup


def split_values(text: str) -> list[str]:
    normalized = text.strip()
    if normalized.startswith("COL "):
        normalized = normalized[4:]

    values: list[str] = []
    for segment in normalized.split(" COL "):
        if " VAL " not in segment:
            continue
        _, value = segment.split(" VAL ", 1)
        values.append(value.strip())
    return values


def to_comem_left(text: str) -> str:
    labels = ["name", "affiliation", "research interests", "papers", "projects"]
    values = (split_values(text) + [""] * 5)[:5]
    return ", ".join(f"{label}: {value}" for label, value in zip(labels, values))


def to_comem_right(text: str) -> str:
    values = (split_values(text) + [""] * 4)[:4]
    labels = ["name", "affiliation", "research interests", "papers", "projects"]
    full_values = [values[0], values[1], "", values[2], values[3]]
    return ", ".join(f"{label}: {value}" for label, value in zip(labels, full_values))


def main() -> None:
    args = parse_args()

    test_df = read_blocking_test(args.blocking_test)

    lookup = build_combined_id_lookup(args.candidates, args.aug_same_name)
    merged = test_df.merge(
        lookup,
        on=["record_left", "record_right", "label"],
        how="left",
        validate="many_to_one",
    )

    if merged[["id_left", "id_right"]].isna().any().any():
        missing = merged[merged["id_left"].isna() | merged["id_right"].isna()].head(5)
        raise ValueError(
            "Some samples could not be backfilled with IDs after checking "
            "candidates and dataset_aug_same_name: "
            f"{missing.to_dict(orient='records')}"
        )

    merged["record_left"] = merged["record_left"].apply(to_comem_left)
    merged["record_right"] = merged["record_right"].apply(to_comem_right)
    merged["label"] = merged["label"].map({"1": "True", "0": "False"})

    output = merged[["id_left", "id_right", "record_left", "record_right", "label"]]
    output = output.reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")

    positive_count = int((output["label"] == "True").sum())
    negative_count = int((output["label"] == "False").sum())
    print(f"test_input_rows={len(test_df)}")
    print(f"output_rows={len(output)}")
    print(f"positive_rows={positive_count}")
    print(f"negative_rows={negative_count}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
