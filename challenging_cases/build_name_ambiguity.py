from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKING_DIR = PROJECT_ROOT / "blocking"

DEFAULT_DATASET_A = BLOCKING_DIR / "dataset_A.csv"
DEFAULT_DATASET_B = BLOCKING_DIR / "dataset_B.csv"
DEFAULT_AUG_A = BLOCKING_DIR / "dataset_A_aug.csv"
DEFAULT_AUG_B = BLOCKING_DIR / "dataset_B_aug.csv"
DEFAULT_MAPPING = BLOCKING_DIR / "A_B_mapping.csv"
DEFAULT_BLOCKING_CANDIDATES = BLOCKING_DIR / "A_B_blocking_candidates.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "challenging_cases" / "name_ambiguity" / "test.txt"

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
            "Build the Name Ambiguity challenging SER test set. The output is a "
            "5-column txt file: id_left, id_right, record_left, record_right, label."
        )
    )
    parser.add_argument("--dataset-a", type=Path, default=DEFAULT_DATASET_A)
    parser.add_argument("--dataset-b", type=Path, default=DEFAULT_DATASET_B)
    parser.add_argument("--aug-a", type=Path, default=DEFAULT_AUG_A)
    parser.add_argument("--aug-b", type=Path, default=DEFAULT_AUG_B)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--blocking-candidates", type=Path, default=DEFAULT_BLOCKING_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-pos", type=int, default=1000)
    parser.add_argument("--num-neg", type=int, default=4000)
    parser.add_argument("--name-sim-threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
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


def extract_attr_field(record: object, field_name: str) -> str:
    text = normalize_text(record)
    pattern = rf"COL\s+{re.escape(field_name)}\s+VAL\s+(.*?)(?=\s+COL\s+|$)"
    match = re.search(pattern, text)
    return normalize_text(match.group(1)) if match else ""


def normalize_name_for_similarity(name: object) -> str:
    text = normalize_text(name).lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def name_similarity(left: object, right: object) -> float:
    left_name = normalize_name_for_similarity(left)
    right_name = normalize_name_for_similarity(right)
    if not left_name or not right_name:
        return 0.0
    return SequenceMatcher(None, left_name, right_name).ratio()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def build_id_to_row(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        normalize_text(row[ID_COL]): row
        for _, row in df.iterrows()
        if normalize_text(row.get(ID_COL, ""))
    }


def build_original_name_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {
        normalize_text(row[ID_COL]): normalize_text(row[NAME_COL])
        for _, row in df.iterrows()
    }


def load_mapped_pairs(mapping_df: pd.DataFrame) -> set[tuple[str, str]]:
    return {
        (normalize_text(row["id_A"]), normalize_text(row["id_B"]))
        for _, row in mapping_df.iterrows()
        if normalize_text(row.get("id_A", "")) and normalize_text(row.get("id_B", ""))
    }


def build_positive_samples(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    id_to_a = build_id_to_row(df_a)
    id_to_b = build_id_to_row(df_b)
    left_columns = [NAME_COL, AFFILIATION_COL, RESEARCH_COL, PAPERS_COL, PROJECTS_COL]
    right_columns = [NAME_COL, AFFILIATION_COL, PAPERS_COL, PROJECTS_COL]

    rows = []
    for _, row in mapping_df.iterrows():
        id_left = normalize_text(row["id_A"])
        id_right = normalize_text(row["id_B"])
        if id_left not in id_to_a or id_right not in id_to_b:
            continue

        rows.append(
            {
                "id_left": id_left,
                "id_right": id_right,
                "record_left": serialize_col_val(id_to_a[id_left], left_columns),
                "record_right": serialize_col_val(id_to_b[id_right], right_columns),
                "label": 1,
            }
        )

    return pd.DataFrame(rows)


def build_negative_samples(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    aug_a: pd.DataFrame,
    aug_b: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    original_name_a = build_original_name_lookup(df_a)
    original_name_b = build_original_name_lookup(df_b)
    mapped_pairs = load_mapped_pairs(mapping_df)

    aug_a = aug_a.copy()
    aug_b = aug_b.copy()
    aug_a["_canonical_name"] = aug_a[ID_COL].map(original_name_a).fillna(aug_a[NAME_COL])
    aug_b["_canonical_name"] = aug_b[ID_COL].map(original_name_b).fillna(aug_b[NAME_COL])
    aug_a["_canonical_name"] = aug_a["_canonical_name"].map(normalize_text)
    aug_b["_canonical_name"] = aug_b["_canonical_name"].map(normalize_text)

    b_by_name = {
        name: group
        for name, group in aug_b.groupby("_canonical_name", sort=False)
        if normalize_text(name)
    }

    left_columns = [NAME_COL, AFFILIATION_COL, RESEARCH_COL, PAPERS_COL, PROJECTS_COL]
    right_columns = [NAME_COL, AFFILIATION_COL, PAPERS_COL, PROJECTS_COL]

    rows = []
    for _, left in aug_a.iterrows():
        name = normalize_text(left["_canonical_name"])
        if not name or name not in b_by_name:
            continue

        id_left = normalize_text(left[ID_COL])
        for _, right in b_by_name[name].iterrows():
            id_right = normalize_text(right[ID_COL])
            if not id_left or not id_right:
                continue
            if (id_left, id_right) in mapped_pairs:
                continue
            if same_affiliation(left[AFFILIATION_COL], right[AFFILIATION_COL]):
                continue

            rows.append(
                {
                    "id_left": id_left,
                    "id_right": id_right,
                    "record_left": serialize_col_val(left, left_columns),
                    "record_right": serialize_col_val(right, right_columns),
                    "label": 0,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["id_left", "id_right", "record_left", "record_right", "label"])

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["id_left", "id_right", "record_left", "record_right"])
        .reset_index(drop=True)
    )


def build_candidate_similar_name_negatives(
    candidates_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    mapped_pairs = load_mapped_pairs(mapping_df)
    rows = []

    for _, row in candidates_df.iterrows():
        if str(row.get("label", "")).strip() != "0":
            continue

        id_left = normalize_text(row.get("id_left", ""))
        id_right = normalize_text(row.get("id_right", ""))
        if not id_left or not id_right:
            continue
        if (id_left, id_right) in mapped_pairs:
            continue

        record_left = normalize_text(row.get("record_left", ""))
        record_right = normalize_text(row.get("record_right", ""))
        left_name = extract_attr_field(record_left, NAME_COL)
        right_name = extract_attr_field(record_right, NAME_COL)
        sim = name_similarity(left_name, right_name)
        if sim < threshold:
            continue

        left_aff = extract_attr_field(record_left, AFFILIATION_COL)
        right_aff = extract_attr_field(record_right, AFFILIATION_COL)
        if same_affiliation(left_aff, right_aff):
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
                "_name_similarity": sim,
                "_blocking_score": score,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["id_left", "id_right", "record_left", "record_right", "label"])

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["id_left", "id_right", "record_left", "record_right"])
        .sort_values(["_name_similarity", "_blocking_score"], ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def sample_exact(df: pd.DataFrame, n: int, seed: int, name: str) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"Not enough {name} samples: need {n}, got {len(df)}")
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def sample_name_ambiguity_negatives(
    strict_negatives: pd.DataFrame,
    similar_negatives: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    strict_negatives = strict_negatives.copy()
    strict_negatives["_source_rank"] = 0

    if len(strict_negatives) >= n:
        return sample_exact(strict_negatives, n, seed, "strict same-name negative")

    existing_keys = set(
        zip(
            strict_negatives["id_left"],
            strict_negatives["id_right"],
            strict_negatives["record_left"],
            strict_negatives["record_right"],
        )
    )
    supplement = similar_negatives.copy()
    if len(supplement):
        supplement["_key"] = list(
            zip(
                supplement["id_left"],
                supplement["id_right"],
                supplement["record_left"],
                supplement["record_right"],
            )
        )
        supplement = supplement[~supplement["_key"].isin(existing_keys)].drop(columns=["_key"])
        supplement["_source_rank"] = 1

    combined = pd.concat([strict_negatives, supplement], ignore_index=True)
    if len(combined) < n:
        raise ValueError(f"Not enough negative samples: need {n}, got {len(combined)}")

    needed = n - len(strict_negatives)
    supplement_sample = sample_exact(supplement, needed, seed, "similar-name negative")
    return (
        pd.concat([strict_negatives, supplement_sample], ignore_index=True)
        .sample(frac=1, random_state=seed + 1)
        .reset_index(drop=True)
    )


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
    aug_a = load_csv(args.aug_a)
    aug_b = load_csv(args.aug_b)
    mapping_df = load_csv(args.mapping)
    candidates_df = load_csv(args.blocking_candidates)

    positives = build_positive_samples(df_a, df_b, mapping_df)
    strict_negatives = build_negative_samples(df_a, df_b, aug_a, aug_b, mapping_df)
    similar_negatives = build_candidate_similar_name_negatives(
        candidates_df=candidates_df,
        mapping_df=mapping_df,
        threshold=args.name_sim_threshold,
    )

    pos_sample = sample_exact(positives, args.num_pos, args.seed, "positive")
    neg_sample = sample_name_ambiguity_negatives(
        strict_negatives=strict_negatives,
        similar_negatives=similar_negatives,
        n=args.num_neg,
        seed=args.seed + 1,
    )

    output_df = (
        pd.concat([pos_sample, neg_sample], ignore_index=True)
        .sample(frac=1, random_state=args.seed + 2)
        .reset_index(drop=True)
    )

    write_finetune_txt(output_df, args.output)

    print(f"positive_pool={len(positives)}")
    print(f"strict_same_name_negative_pool={len(strict_negatives)}")
    print(f"similar_name_negative_pool={len(similar_negatives)}")
    print(f"output_rows={len(output_df)}")
    print(f"label_counts={output_df['label'].value_counts().sort_index().to_dict()}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
