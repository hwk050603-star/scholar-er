from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKING_DIR = PROJECT_ROOT / "blocking"
PRETRAIN_DATASET_DIR = PROJECT_ROOT / "pretrain" / "dataset"

DEFAULT_DATASET_A = BLOCKING_DIR / "dataset_A.csv"
DEFAULT_DATASET_B = BLOCKING_DIR / "dataset_B.csv"
DEFAULT_MAPPING = BLOCKING_DIR / "A_B_mapping.csv"
DEFAULT_BLOCKING_CANDIDATES = BLOCKING_DIR / "A_B_blocking_candidates.csv"
DEFAULT_PARTIAL_WORK_HISTORY = PRETRAIN_DATASET_DIR / "partial_work_history.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "challenging_cases" / "affiliation_shifts" / "test.txt"

ID_COL = "Id"
NAME_COL = "Name"
AFFILIATION_COL = "Affiliation"
RESEARCH_COL = "Research Interests"
PAPERS_COL = "Papers"
PROJECTS_COL = "Projects"

COMMON_UNI_ABBR = {
    "中国科学院": "中科院",
    "中国人民大学": "人大",
    "华中科技大学": "华科",
    "浙江工业大学": "浙工大",
    "中国农业大学": "中国农大",
    "上海理工大学": "上理工",
    "江苏师范大学": "江苏师大",
    "南通大学": "南通大",
    "宁波大学": "宁大",
    "青海大学": "青大",
}

ORG_ROOT_KEYWORDS = [
    "大学",
    "科学院",
    "社会科学院",
    "工程院",
]

ORG_FALLBACK_KEYWORDS = [
    "研究所",
    "研究院",
    "研究中心",
    "重点实验室",
    "实验室",
    "学院",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Affiliation Shifts challenging SER test set. Positive "
            "pairs use real A/B mapped records, while replacing the B-side "
            "affiliation with another historical affiliation of the same scholar."
        )
    )
    parser.add_argument("--dataset-a", type=Path, default=DEFAULT_DATASET_A)
    parser.add_argument("--dataset-b", type=Path, default=DEFAULT_DATASET_B)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--blocking-candidates", type=Path, default=DEFAULT_BLOCKING_CANDIDATES)
    parser.add_argument("--partial-work-history", type=Path, default=DEFAULT_PARTIAL_WORK_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-pos", type=int, default=1000)
    parser.add_argument("--num-neg", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def strip_time_from_segment(seg: object) -> str:
    text = normalize_text(seg)
    if not text:
        return ""
    return re.sub(r"[（(][^（）()]{0,60}[)）]\s*$", "", text).strip()


def parse_timeline_text(timeline_text: object) -> list[str]:
    text = normalize_text(timeline_text)
    if not text:
        return []
    seen = set()
    out = []
    for part in text.split("|"):
        aff = strip_time_from_segment(part)
        if aff and aff not in seen:
            seen.add(aff)
            out.append(aff)
    return out


def clean_org(org: object) -> str:
    text = normalize_text(org)
    text = re.sub(r"[（(].*?[）)]", "", text)
    text = re.sub(r"[、,，;；/]+", " ", text)
    return normalize_text(text)


def extract_main_org(org: object) -> str:
    text = clean_org(org)
    if not text:
        return ""

    for full, abbr in COMMON_UNI_ABBR.items():
        text = text.replace(abbr, full)

    for keyword in ORG_ROOT_KEYWORDS:
        idx = text.find(keyword)
        if idx != -1:
            return text[: idx + len(keyword)].strip()

    for keyword in ORG_FALLBACK_KEYWORDS:
        idx = text.find(keyword)
        if idx != -1:
            return text[: idx + len(keyword)].strip()
    return text


def affiliation_matches(current_org: object, timeline_aff: object) -> bool:
    current = normalize_text(current_org)
    aff = normalize_text(timeline_aff)
    if not current or not aff:
        return False
    if current in aff or aff in current:
        return True
    current_main = extract_main_org(current)
    aff_main = extract_main_org(aff)
    return bool(current_main and aff_main and current_main == aff_main)


def is_affiliation_shift(current_org: object, candidate_aff: object) -> bool:
    current_main = extract_main_org(current_org)
    candidate_main = extract_main_org(candidate_aff)
    if not current_main or not candidate_main:
        return False
    if current_main == candidate_main:
        return False
    return not affiliation_matches(current_org, candidate_aff)


def serialize_left_record(row: pd.Series) -> str:
    columns = [NAME_COL, AFFILIATION_COL, RESEARCH_COL, PAPERS_COL, PROJECTS_COL]
    return " ".join(
        f"COL {column} VAL {normalize_text(row.get(column, ''))}"
        for column in columns
    ).strip()


def serialize_right_record(row: pd.Series, override_affiliation: str | None = None) -> str:
    columns = [NAME_COL, AFFILIATION_COL, PAPERS_COL, PROJECTS_COL]
    return " ".join(
        f"COL {column} VAL "
        f"{normalize_text(override_affiliation if column == AFFILIATION_COL and override_affiliation is not None else row.get(column, ''))}"
        for column in columns
    ).strip()


def extract_attr_field(record: object, field_name: str) -> str:
    text = normalize_text(record)
    pattern = rf"COL\s+{re.escape(field_name)}\s+VAL\s+(.*?)(?=\s+COL\s+|$)"
    match = re.search(pattern, text)
    return normalize_text(match.group(1)) if match else ""


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def build_id_to_row(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        normalize_text(row[ID_COL]): row
        for _, row in df.iterrows()
        if normalize_text(row.get(ID_COL, ""))
    }


def load_partial_work_history(path: Path) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    rows = []
    name_to_rows: dict[str, list[dict[str, object]]] = defaultdict(list)

    with open(path, "r", encoding="utf-8-sig") as f:
        header = f.readline().strip().lstrip("\ufeff").split(",")
        if len(header) < 3 or header[:3] != ["Id", "Name", "Work History"]:
            raise ValueError(f"Unexpected partial_work_history.csv header: {','.join(header)}")

        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                parts += [""] * (3 - len(parts))
            item = {
                "timeline_id": normalize_text(parts[0]),
                "name": normalize_text(parts[1]),
                "timeline_text": normalize_text(parts[2]),
                "affiliations": parse_timeline_text(parts[2]),
            }
            rows.append(item)
            name_to_rows[item["name"]].append(item)

    return rows, name_to_rows


def choose_matched_timeline_record(
    name: str,
    current_org: str,
    name_to_timeline_rows: dict[str, list[dict[str, object]]],
) -> dict[str, object] | None:
    for item in name_to_timeline_rows.get(normalize_text(name), []):
        if any(affiliation_matches(current_org, aff) for aff in item.get("affiliations", [])):
            return item
    return None


def build_positive_samples(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    mapping_df: pd.DataFrame,
    name_to_timeline_rows: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    id_to_a = build_id_to_row(df_a)
    id_to_b = build_id_to_row(df_b)
    rows = []

    for _, mapping_row in mapping_df.iterrows():
        id_left = normalize_text(mapping_row.get("id_A", ""))
        id_right = normalize_text(mapping_row.get("id_B", ""))
        if id_left not in id_to_a or id_right not in id_to_b:
            continue

        left = id_to_a[id_left]
        right = id_to_b[id_right]
        name = normalize_text(left.get(NAME_COL, ""))
        current_org = normalize_text(left.get(AFFILIATION_COL, ""))
        right_current_org = normalize_text(right.get(AFFILIATION_COL, ""))
        if not name or not current_org or not right_current_org:
            continue

        timeline_item = choose_matched_timeline_record(name, current_org, name_to_timeline_rows)
        if timeline_item is None:
            continue

        for candidate_aff in timeline_item.get("affiliations", []):
            if not is_affiliation_shift(current_org, candidate_aff):
                continue
            if not is_affiliation_shift(right_current_org, candidate_aff):
                continue
            rows.append(
                {
                    "id_left": id_left,
                    "id_right": id_right,
                    "record_left": serialize_left_record(left),
                    "record_right": serialize_right_record(right, override_affiliation=candidate_aff),
                    "label": 1,
                }
            )

    return pd.DataFrame(rows).drop_duplicates(
        subset=["id_left", "id_right", "record_left", "record_right"]
    ).reset_index(drop=True)


def build_negative_samples(candidates_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidates_df.iterrows():
        if str(row.get("label", "")).strip() != "0":
            continue

        id_left = normalize_text(row.get("id_left", ""))
        id_right = normalize_text(row.get("id_right", ""))
        record_left = normalize_text(row.get("record_left", ""))
        record_right = normalize_text(row.get("record_right", ""))
        if not id_left or not id_right or not record_left or not record_right:
            continue

        left_aff = extract_attr_field(record_left, AFFILIATION_COL)
        right_aff = extract_attr_field(record_right, AFFILIATION_COL)
        if not is_affiliation_shift(left_aff, right_aff):
            continue

        try:
            score = float(row.get("score", 0.0))
        except ValueError:
            score = 0.0

        rows.append(
            {
                "id_left": id_left,
                "id_right": id_right,
                "record_left": record_left,
                "record_right": record_right,
                "label": 0,
                "_score": score,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["id_left", "id_right", "record_left", "record_right", "label"])

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["id_left", "id_right", "record_left", "record_right"])
        .sort_values("_score", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def sample_exact(df: pd.DataFrame, n: int, seed: int, name: str) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Not enough {name} samples: need {n}, got {len(df)}")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def select_hard_negatives(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Not enough negative samples: need {n}, got {len(df)}")
    return df.head(n).reset_index(drop=True)


def write_finetune_txt(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["id_left", "id_right", "record_left", "record_right", "label"]].to_csv(
        output_path,
        sep="\t",
        index=False,
        header=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    args = parse_args()

    df_a = load_csv(args.dataset_a)
    df_b = load_csv(args.dataset_b)
    mapping_df = load_csv(args.mapping)
    candidates_df = load_csv(args.blocking_candidates)
    _, name_to_timeline_rows = load_partial_work_history(args.partial_work_history)

    positives = build_positive_samples(df_a, df_b, mapping_df, name_to_timeline_rows)
    negatives = build_negative_samples(candidates_df)

    pos_sample = sample_exact(positives, args.num_pos, args.seed, "positive")
    neg_sample = select_hard_negatives(negatives, args.num_neg)

    output_df = (
        pd.concat([pos_sample, neg_sample], ignore_index=True)
        .sample(frac=1, random_state=args.seed + 2)
        .reset_index(drop=True)
    )
    write_finetune_txt(output_df, args.output)

    print(f"positive_pool={len(positives)}")
    print(f"negative_pool={len(negatives)}")
    print(f"output_rows={len(output_df)}")
    print(f"label_counts={output_df['label'].value_counts().sort_index().to_dict()}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
