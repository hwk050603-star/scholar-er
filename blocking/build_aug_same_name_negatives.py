from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_A = SCRIPT_DIR / "dataset_A.csv"
DEFAULT_DATASET_B = SCRIPT_DIR / "dataset_B.csv"
DEFAULT_AUG_A = SCRIPT_DIR / "dataset_A_aug.csv"
DEFAULT_AUG_B = SCRIPT_DIR / "dataset_B_aug.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset_aug_same_name.csv"

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

ORG_SUFFIXES = [
    "全国重点实验室",
    "国家重点实验室",
    "重点实验室",
    "实验室",
    "研究中心",
    "研究院",
    "研究所",
    "学院",
    "学部",
    "中心",
    "系",
    "处",
]

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
            "Build same-name negative pairs from augmented dataset_A/B, "
            "treating pinyin name variants and organization abbreviations as equivalent."
        )
    )
    parser.add_argument("--dataset-a", type=Path, default=DEFAULT_DATASET_A)
    parser.add_argument("--dataset-b", type=Path, default=DEFAULT_DATASET_B)
    parser.add_argument("--aug-a", type=Path, default=DEFAULT_AUG_A)
    parser.add_argument("--aug-b", type=Path, default=DEFAULT_AUG_B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def clean_org(org: object) -> str:
    text = normalize_text(org)
    text = re.sub(r"[（(].*?[）)]", "", text)
    text = re.sub(r"[、,，;；/]+", " ", text)
    return normalize_text(text)


def extract_main_org(org: object) -> str:
    text = clean_org(org)
    if not text:
        return ""

    for keyword in ORG_ROOT_KEYWORDS:
        idx = text.find(keyword)
        if idx != -1:
            return text[: idx + len(keyword)].strip()

    for keyword in ORG_FALLBACK_KEYWORDS:
        idx = text.find(keyword)
        if idx != -1:
            return text[: idx + len(keyword)].strip()
    return text


def build_org_abbr(org: object) -> str:
    text = clean_org(org)
    if not text:
        return ""

    main_org = extract_main_org(text)
    for full, abbr in COMMON_UNI_ABBR.items():
        if full in main_org:
            return main_org.replace(full, abbr, 1)
        if full in text:
            return abbr

    if any(keyword in main_org for keyword in ORG_ROOT_KEYWORDS):
        return main_org

    trimmed = main_org
    for suffix in ORG_SUFFIXES:
        if trimmed.endswith(suffix) and len(trimmed) > len(suffix):
            trimmed = trimmed[: -len(suffix)].strip()
            break
    return trimmed or main_org


def org_variants(org: object) -> set[str]:
    variants = {clean_org(org)}
    variants.add(extract_main_org(org))
    variants.add(build_org_abbr(org))
    variants.add(build_org_abbr(extract_main_org(org)))

    cleaned = clean_org(org)
    for full, abbr in COMMON_UNI_ABBR.items():
        if cleaned.startswith(full) or cleaned.startswith(abbr):
            variants.add(full)
            variants.add(abbr)

    return {variant for variant in variants if variant}


def same_affiliation(left: object, right: object) -> bool:
    left_variants = org_variants(left)
    right_variants = org_variants(right)
    return bool(left_variants & right_variants)


def serialize_col_val(row: pd.Series, columns: list[str]) -> str:
    return " ".join(
        f"COL {column} VAL {normalize_text(row.get(column, ''))}"
        for column in columns
    ).strip()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def build_original_name_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {
        normalize_text(row[ID_COL]): normalize_text(row[NAME_COL])
        for _, row in df.iterrows()
    }


def main() -> None:
    args = parse_args()

    df_a = load_csv(args.dataset_a)
    df_b = load_csv(args.dataset_b)
    aug_a = load_csv(args.aug_a)
    aug_b = load_csv(args.aug_b)

    original_name_a = build_original_name_lookup(df_a)
    original_name_b = build_original_name_lookup(df_b)

    aug_a = aug_a.copy()
    aug_b = aug_b.copy()
    aug_a["_canonical_name"] = aug_a[ID_COL].map(original_name_a).fillna(aug_a[NAME_COL])
    aug_b["_canonical_name"] = aug_b[ID_COL].map(original_name_b).fillna(aug_b[NAME_COL])
    aug_a["_canonical_name"] = aug_a["_canonical_name"].map(normalize_text)
    aug_b["_canonical_name"] = aug_b["_canonical_name"].map(normalize_text)

    b_by_name: dict[str, pd.DataFrame] = {
        name: group
        for name, group in aug_b.groupby("_canonical_name", sort=False)
        if normalize_text(name)
    }

    left_columns = [
        NAME_COL,
        AFFILIATION_COL,
        RESEARCH_COL,
        PAPERS_COL,
        PROJECTS_COL,
    ]
    right_columns = [NAME_COL, AFFILIATION_COL, PAPERS_COL, PROJECTS_COL]

    rows: list[dict[str, object]] = []
    per_name_counts: defaultdict[str, int] = defaultdict(int)
    for _, left in aug_a.iterrows():
        name = normalize_text(left["_canonical_name"])
        if not name or name not in b_by_name:
            continue

        for _, right in b_by_name[name].iterrows():
            if same_affiliation(left[AFFILIATION_COL], right[AFFILIATION_COL]):
                continue

            rows.append(
                {
                    "id_left": normalize_text(left[ID_COL]),
                    "id_right": normalize_text(right[ID_COL]),
                    "record_left": serialize_col_val(left, left_columns),
                    "record_right": serialize_col_val(right, right_columns),
                    "label": 0,
                }
            )
            per_name_counts[name] += 1

    output_df = pd.DataFrame(
        rows,
        columns=["id_left", "id_right", "record_left", "record_right", "label"],
    ).drop_duplicates(subset=["id_left", "id_right"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"dataset_a_rows={len(df_a)}")
    print(f"dataset_b_rows={len(df_b)}")
    print(f"aug_a_rows={len(aug_a)}")
    print(f"aug_b_rows={len(aug_b)}")
    print(f"same_name_diff_affiliation_rows={len(output_df)}")
    print(f"matched_names={len(per_name_counts)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
