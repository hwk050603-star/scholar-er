from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from pypinyin import lazy_pinyin
except ImportError:
    lazy_pinyin = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if (REPO_ROOT / "vendor_src").exists():
    REPO_ROOT = REPO_ROOT
else:
    REPO_ROOT = SCRIPT_DIR.parent.parent

LOCAL_RETRIV_SRC = (REPO_ROOT / "vendor_src" / "retriv-0.2.3").resolve()
if str(LOCAL_RETRIV_SRC) not in sys.path and LOCAL_RETRIV_SRC.exists():
    sys.path.insert(0, str(LOCAL_RETRIV_SRC))

SparseRetriever = None
set_base_path = None
nltk = None


ID_COLUMN = "Id"
NAME_COLUMN = "Name"
AFFILIATION_COLUMN = "Affiliation"
RESEARCH_COLUMN = "Research Interests"
PAPERS_COLUMN = "Papers"
PROJECTS_COLUMN = "Projects"

FIELD_ALIASES = {
    "序号": ID_COLUMN,
    "姓名": NAME_COLUMN,
    "单位": AFFILIATION_COLUMN,
    "研究领域": RESEARCH_COLUMN,
    "论文": PAPERS_COLUMN,
    "项目": PROJECTS_COLUMN,
}

DEFAULT_A_COLUMNS = [
    NAME_COLUMN,
    AFFILIATION_COLUMN,
    RESEARCH_COLUMN,
    PAPERS_COLUMN,
    PROJECTS_COLUMN,
]
DEFAULT_B_COLUMNS = [NAME_COLUMN, AFFILIATION_COLUMN, PAPERS_COLUMN, PROJECTS_COLUMN]
DEFAULT_WEIGHTS = {
    NAME_COLUMN: 3.0,
    AFFILIATION_COLUMN: 2.0,
    RESEARCH_COLUMN: 1.5,
    PAPERS_COLUMN: 1.0,
    PROJECTS_COLUMN: 1.0,
}

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

INDEX_DIR = SCRIPT_DIR / ".retriv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sparkly-style sparse blocking for dataset_A.csv and dataset_B.csv."
    )
    parser.add_argument("--left", default=str(SCRIPT_DIR / "dataset_A.csv"))
    parser.add_argument("--right", default=str(SCRIPT_DIR / "dataset_B.csv"))
    parser.add_argument("--aug-left", default=str(SCRIPT_DIR / "dataset_A_aug.csv"))
    parser.add_argument("--aug-right", default=str(SCRIPT_DIR / "dataset_B_aug.csv"))
    parser.add_argument("--mapping", default=str(SCRIPT_DIR / "A_B_mapping.csv"))
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "A_B_blocking_candidates.csv"),
    )
    parser.add_argument("--dataset-name", default="dataset-a-dataset-b")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--weights",
        default="Name=3,Affiliation=2,Research Interests=1.5,Papers=1,Projects=1",
        help="Column weights encoded by repeating COL/VAL segments in the text.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild SparseRetriever index instead of loading an existing one.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Use --left/--right directly instead of generating augmented datasets first.",
    )
    parser.add_argument("--name-chinese-pinyin-ratio", type=float, default=0.1)
    parser.add_argument("--name-western-pinyin-ratio", type=float, default=0.1)
    parser.add_argument("--name-paper-ratio", type=float, default=0.1)
    parser.add_argument("--org-clean-ratio", type=float, default=0.1)
    parser.add_argument("--org-main-ratio", type=float, default=0.1)
    parser.add_argument("--org-abbr-ratio", type=float, default=0.1)
    parser.add_argument("--field-missing-ratio", type=float, default=0.05)
    parser.add_argument("--dirty-data-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_weights(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not raw.strip():
        return DEFAULT_WEIGHTS.copy()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid weight spec: {item}")
        key, value = item.split("=", 1)
        key = FIELD_ALIASES.get(key.strip(), key.strip())
        weights[key] = float(value.strip())
    return weights


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=FIELD_ALIASES)


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return " ".join(text.split())


def clean_name(name: str) -> str:
    return re.sub(r"[（(].*?[）)]", "", normalize_text(name)).strip()


def split_chinese_name(name: str) -> tuple[str, str]:
    name = clean_name(name)
    if len(name) < 2:
        return name, ""
    return name[0], name[1:]


def to_chinese_pinyin(name: str) -> str:
    name = clean_name(name)
    if not name or lazy_pinyin is None:
        return name
    surname, given = split_chinese_name(name)
    if not given:
        return "".join(lazy_pinyin(name)).capitalize()
    surname_py = "".join(lazy_pinyin(surname)).capitalize()
    given_py = "".join(lazy_pinyin(given)).capitalize()
    return f"{surname_py} {given_py}".strip()


def to_western_pinyin(name: str) -> str:
    name = clean_name(name)
    if not name or lazy_pinyin is None:
        return name
    surname, given = split_chinese_name(name)
    if not given:
        return "".join(lazy_pinyin(name)).capitalize()
    surname_py = "".join(lazy_pinyin(surname)).capitalize()
    given_py = "".join(lazy_pinyin(given)).capitalize()
    return f"{given_py} {surname_py}".strip()


def to_paper_style(name: str) -> str:
    name = clean_name(name)
    if not name or lazy_pinyin is None:
        return name
    surname, given = split_chinese_name(name)
    if not given:
        return "".join(lazy_pinyin(name)).capitalize()
    surname_py = "".join(lazy_pinyin(surname)).capitalize()
    given_py = "".join(lazy_pinyin(given)).capitalize()
    return f"{given_py} {surname_py[:1]}.".strip()


def clean_org(org: str) -> str:
    org = normalize_text(org)
    org = re.sub(r"[（(].*?[）)]", "", org)
    org = re.sub(r"[、,，;；]+", " ", org)
    return re.sub(r"\s+", " ", org).strip()


def extract_main_org(org: str) -> str:
    org = clean_org(org)
    if not org:
        return ""

    # Keep the university / academy level when a lower-level school,
    # department, lab, or center follows it.
    for keyword in ORG_ROOT_KEYWORDS:
        idx = org.find(keyword)
        if idx != -1:
            return org[: idx + len(keyword)].strip()

    for keyword in ORG_FALLBACK_KEYWORDS:
        idx = org.find(keyword)
        if idx != -1:
            return org[: idx + len(keyword)].strip()
    return org


def build_org_abbr(org: str) -> str:
    org = clean_org(org)
    if not org:
        return ""

    main_org = extract_main_org(org)
    for full, abbr in COMMON_UNI_ABBR.items():
        if full in main_org:
            return main_org.replace(full, abbr, 1)
        if full in org:
            return abbr

    if any(keyword in main_org for keyword in ORG_ROOT_KEYWORDS):
        return main_org

    trimmed = main_org
    for suffix in ORG_SUFFIXES:
        if trimmed.endswith(suffix) and len(trimmed) > len(suffix):
            trimmed = trimmed[: -len(suffix)].strip()
            break
    return trimmed or main_org


def choose_indices(length: int, ratio: float, rng: random.Random, used: set[int]) -> list[int]:
    if ratio <= 0:
        return []
    available = [i for i in range(length) if i not in used]
    count = min(len(available), int(round(length * ratio)))
    if count <= 0:
        return []
    chosen = rng.sample(available, count)
    used.update(chosen)
    return chosen


def mutable_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column != ID_COLUMN and column in {
            NAME_COLUMN,
            AFFILIATION_COLUMN,
            RESEARCH_COLUMN,
            PAPERS_COLUMN,
            PROJECTS_COLUMN,
        }
    ]


def choose_non_empty_column(row: pd.Series, columns: list[str], rng: random.Random) -> str | None:
    available = [column for column in columns if normalize_text(row.get(column, ""))]
    if not available:
        return None
    return rng.choice(available)


def inject_dirty_text(value: object, rng: random.Random) -> str:
    text = normalize_text(value)
    if not text:
        return text

    operation = rng.choice(["delete", "duplicate", "swap", "separator", "append"])
    if operation == "delete" and len(text) > 1:
        idx = rng.randrange(len(text))
        return text[:idx] + text[idx + 1 :]
    if operation == "duplicate":
        idx = rng.randrange(len(text))
        return text[:idx] + text[idx] + text[idx:]
    if operation == "swap" and len(text) > 1:
        idx = rng.randrange(len(text) - 1)
        chars = list(text)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    if operation == "separator":
        if "|" in text:
            return text.replace("|", " ", 1)
        if " " in text:
            return text.replace(" ", "", 1)
    return f"{text} 未明确"


def augment_df(
    df: pd.DataFrame,
    name_chinese_pinyin_ratio: float,
    name_western_pinyin_ratio: float,
    name_paper_ratio: float,
    org_clean_ratio: float,
    org_main_ratio: float,
    org_abbr_ratio: float,
    field_missing_ratio: float,
    dirty_data_ratio: float,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    rng = random.Random(seed)
    length = len(out)

    used_name: set[int] = set()
    chinese_idx = choose_indices(length, name_chinese_pinyin_ratio, rng, used_name)
    western_idx = choose_indices(length, name_western_pinyin_ratio, rng, used_name)
    paper_idx = choose_indices(length, name_paper_ratio, rng, used_name)

    used_org: set[int] = set()
    clean_idx = choose_indices(length, org_clean_ratio, rng, used_org)
    main_idx = choose_indices(length, org_main_ratio, rng, used_org)
    abbr_idx = choose_indices(length, org_abbr_ratio, rng, used_org)

    columns_for_noise = mutable_columns(out)
    missing_idx = choose_indices(length, field_missing_ratio, rng, set())
    dirty_idx = choose_indices(length, dirty_data_ratio, rng, set())

    for idx in chinese_idx:
        out.at[idx, NAME_COLUMN] = to_chinese_pinyin(out.iloc[idx][NAME_COLUMN])
    for idx in western_idx:
        out.at[idx, NAME_COLUMN] = to_western_pinyin(out.iloc[idx][NAME_COLUMN])
    for idx in paper_idx:
        out.at[idx, NAME_COLUMN] = to_paper_style(out.iloc[idx][NAME_COLUMN])

    for idx in clean_idx:
        out.at[idx, AFFILIATION_COLUMN] = clean_org(out.iloc[idx][AFFILIATION_COLUMN])
    for idx in main_idx:
        out.at[idx, AFFILIATION_COLUMN] = extract_main_org(out.iloc[idx][AFFILIATION_COLUMN])
    for idx in abbr_idx:
        out.at[idx, AFFILIATION_COLUMN] = build_org_abbr(out.iloc[idx][AFFILIATION_COLUMN])

    for idx in missing_idx:
        if not columns_for_noise:
            continue
        column = rng.choice(columns_for_noise)
        out.at[idx, column] = ""

    for idx in dirty_idx:
        column = choose_non_empty_column(out.iloc[idx], columns_for_noise, rng)
        if column is None:
            continue
        out.at[idx, column] = inject_dirty_text(out.iloc[idx][column], rng)

    return out


def write_augmented_datasets(args: argparse.Namespace) -> tuple[str, str]:
    df_a = normalize_columns(pd.read_csv(args.left, dtype=str).fillna(""))
    df_b = normalize_columns(pd.read_csv(args.right, dtype=str).fillna(""))

    aug_a = augment_df(
        df_a,
        name_chinese_pinyin_ratio=args.name_chinese_pinyin_ratio,
        name_western_pinyin_ratio=args.name_western_pinyin_ratio,
        name_paper_ratio=args.name_paper_ratio,
        org_clean_ratio=args.org_clean_ratio,
        org_main_ratio=args.org_main_ratio,
        org_abbr_ratio=args.org_abbr_ratio,
        field_missing_ratio=args.field_missing_ratio,
        dirty_data_ratio=args.dirty_data_ratio,
        seed=args.seed,
    )
    aug_b = augment_df(
        df_b,
        name_chinese_pinyin_ratio=args.name_chinese_pinyin_ratio,
        name_western_pinyin_ratio=args.name_western_pinyin_ratio,
        name_paper_ratio=args.name_paper_ratio,
        org_clean_ratio=args.org_clean_ratio,
        org_main_ratio=args.org_main_ratio,
        org_abbr_ratio=args.org_abbr_ratio,
        field_missing_ratio=args.field_missing_ratio,
        dirty_data_ratio=args.dirty_data_ratio,
        seed=args.seed,
    )

    aug_a = aug_a[df_a.columns]
    aug_b = aug_b[df_b.columns]

    aug_left = Path(args.aug_left)
    aug_right = Path(args.aug_right)
    aug_left.parent.mkdir(parents=True, exist_ok=True)
    aug_right.parent.mkdir(parents=True, exist_ok=True)
    aug_a.to_csv(aug_left, index=False, encoding="utf-8-sig")
    aug_b.to_csv(aug_right, index=False, encoding="utf-8-sig")

    print(f"Saved augmented left: {aug_left} ({len(aug_a)} rows)")
    print(f"Saved augmented right: {aug_right} ({len(aug_b)} rows)")
    if lazy_pinyin is None:
        print("Warning: pypinyin not installed, name augmentation falls back to original values.")
    return str(aug_left), str(aug_right)


def serialize_col_val(row: pd.Series, columns: list[str]) -> str:
    parts: list[str] = []
    for column in columns:
        value = normalize_text(row.get(column, ""))
        parts.append(f"COL {column} VAL {value}")
    return " ".join(parts).strip()


def serialize_weighted_col_val(
    row: pd.Series,
    columns: list[str],
    weights: dict[str, float],
) -> str:
    parts: list[str] = []
    for column in columns:
        value = normalize_text(row.get(column, ""))
        segment = f"COL {column} VAL {value}"
        repeat = max(1, int(round(weights.get(column, 1.0))))
        parts.extend([segment] * repeat)
    return " ".join(parts).strip()


def build_record_df(
    path: str,
    columns: list[str],
    weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(pd.read_csv(path, dtype=str).fillna(""))
    df[ID_COLUMN] = df[ID_COLUMN].astype(int)
    df["record"] = df.apply(lambda row: serialize_col_val(row, columns), axis=1)
    df["blocking_text"] = df.apply(
        lambda row: serialize_weighted_col_val(row, columns, weights),
        axis=1,
    )
    record_df = df.set_index(ID_COLUMN)[["record", "blocking_text"]].copy()
    return df, record_df


def generate_docs(df: pd.DataFrame):
    for idx, row in df.iterrows():
        text = row["blocking_text"] if "blocking_text" in df.columns else " ".join(map(str, row.values))
        yield {
            "id": idx,
            "text": text,
        }


def get_index_meta_path(index_name: str) -> Path:
    return INDEX_DIR / f"{index_name}.meta.json"


def build_index_meta(
    right_path: str,
    columns: list[str],
    weights: dict[str, float],
) -> dict[str, object]:
    resolved = Path(right_path).resolve()
    stat = resolved.stat()
    return {
        "right_path": str(resolved),
        "right_mtime_ns": stat.st_mtime_ns,
        "right_size": stat.st_size,
        "columns": columns,
        "weights": weights,
        "tokenizer": "whitespace",
        "stemmer": None,
        "stopwords": None,
    }


def load_index_meta(index_name: str) -> dict[str, object] | None:
    meta_path = get_index_meta_path(index_name)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_index_meta(index_name: str, meta: dict[str, object]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = get_index_meta_path(index_name)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def should_rebuild_index(
    index_name: str,
    right_path: str,
    columns: list[str],
    weights: dict[str, float],
) -> bool:
    current_meta = build_index_meta(right_path, columns, weights)
    saved_meta = load_index_meta(index_name)
    return saved_meta != current_meta


def sparkly_blocking(
    dataset_name: str,
    right_path: str,
    right_columns: list[str],
    weights: dict[str, float],
    left_record_df: pd.DataFrame,
    right_record_df: pd.DataFrame,
    matches: set[tuple[int, int]],
    topk: int,
    rebuild_index: bool,
) -> list[tuple[int, int, float]]:
    global SparseRetriever, nltk, set_base_path
    if SparseRetriever is None:
        try:
            import nltk as nltk_module
            from retriv import SparseRetriever as SparseRetrieverClass
            from retriv import set_base_path as set_base_path_func
        except ImportError:
            nltk_module = None
            SparseRetrieverClass = None
            set_base_path_func = None
        nltk = nltk_module
        SparseRetriever = SparseRetrieverClass
        set_base_path = set_base_path_func

    if SparseRetriever is None:
        raise ImportError(
            "retriv is not installed. Install retriv to run Sparkly-style blocking."
        )

    if nltk is not None:
        nltk.download = lambda *args, **kwargs: None
    if set_base_path is not None:
        set_base_path(str(INDEX_DIR.resolve()))

    index_name = f"{dataset_name}-index"
    current_meta = build_index_meta(right_path, right_columns, weights)
    auto_rebuild = should_rebuild_index(index_name, right_path, right_columns, weights)
    if rebuild_index or auto_rebuild:
        retriever = SparseRetriever(
            index_name=index_name,
            tokenizer="whitespace",
            stemmer=None,
            stopwords=None,
        )
        retriever = retriever.index(generate_docs(right_record_df), show_progress=True)
        save_index_meta(index_name, current_meta)
    else:
        try:
            retriever = SparseRetriever.load(index_name)
        except FileNotFoundError:
            retriever = SparseRetriever(
                index_name=index_name,
                tokenizer="whitespace",
                stemmer=None,
                stopwords=None,
            )
            retriever = retriever.index(generate_docs(right_record_df), show_progress=True)
            save_index_meta(index_name, current_meta)

    queries = list(generate_docs(left_record_df))
    candidates = retriever.bsearch(queries, show_progress=True, cutoff=topk)

    candidates_k: list[tuple[int, int, float]] = []
    for left_id, right_scores in candidates.items():
        for right_id in sorted(right_scores, key=right_scores.get, reverse=True):
            candidates_k.append((left_id, right_id, float(right_scores[right_id])))

    candidate_pairs = {(left_id, right_id) for left_id, right_id, _ in candidates_k}
    recall = len(matches & candidate_pairs) / len(matches) * 100 if matches else 0.0
    print(f"Recall@{topk}: {recall:.2f}")
    return candidates_k


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.weights)

    left_columns = DEFAULT_A_COLUMNS
    right_columns = DEFAULT_B_COLUMNS

    if args.no_augment:
        left_path = args.left
        right_path = args.right
    else:
        left_path, right_path = write_augmented_datasets(args)

    left_df, left_record_df = build_record_df(left_path, left_columns, weights)
    right_df, right_record_df = build_record_df(right_path, right_columns, weights)
    mapping_df = pd.read_csv(args.mapping)

    matches = {
        (int(row.id_A), int(row.id_B))
        for row in mapping_df.itertuples(index=False)
    }

    candidates = sparkly_blocking(
        dataset_name=args.dataset_name,
        right_path=right_path,
        right_columns=right_columns,
        weights=weights,
        left_record_df=left_record_df,
        right_record_df=right_record_df,
        matches=matches,
        topk=args.topk,
        rebuild_index=args.rebuild_index,
    )

    left_record_map = left_record_df["record"].to_dict()
    right_record_map = right_record_df["record"].to_dict()

    output_rows = []
    for left_id, right_id, score in candidates:
        output_rows.append(
            {
                "id_left": left_id,
                "id_right": right_id,
                "score": score,
                "record_left": left_record_map[left_id],
                "record_right": right_record_map[right_id],
                "label": int((left_id, right_id) in matches),
            }
        )

    output_df = pd.DataFrame(output_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved candidates to: {output_path}")
    print(f"Rows: {len(output_df)}")


if __name__ == "__main__":
    main()
