from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLOCKING_DIR = PROJECT_ROOT / "blocking"

DEFAULT_CANDIDATES = BLOCKING_DIR / "A_B_blocking_candidates.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "challenging_cases" / "missing_attributes" / "test.txt"

RESEARCH_COL = "Research Interests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Missing Attributes challenging SER test set. The script "
            "samples real A/B candidate pairs and masks research interests."
        )
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-pos", type=int, default=1000)
    parser.add_argument("--num-neg", type=int, default=4000)
    parser.add_argument("--mask-side", choices=["left", "right", "both"], default="left")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", text)


def mask_col_val_field(record: object, field_name: str) -> str:
    text = normalize_text(record)
    if not text:
        return text

    pattern = rf"\s*COL\s+{re.escape(field_name)}\s+VAL\s+.*?(?=\s+COL\s+|$)"
    if re.search(pattern, text):
        return normalize_text(re.sub(pattern, " ", text))
    return text


def load_candidates(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    df["label"] = df["label"].astype(str).str.strip()
    df["_score"] = pd.to_numeric(df.get("score", 0.0), errors="coerce").fillna(0.0)
    return df


def prepare_samples(df: pd.DataFrame) -> pd.DataFrame:
    required = ["id_left", "id_right", "record_left", "record_right", "label"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in candidates file: {missing}")

    out = df[required + ["_score"]].copy()
    for column in required:
        out[column] = out[column].map(normalize_text)
    out = out[
        (out["id_left"] != "")
        & (out["id_right"] != "")
        & (out["record_left"] != "")
        & (out["record_right"] != "")
        & (out["label"].isin(["0", "1"]))
    ].copy()
    return out.drop_duplicates(
        subset=["id_left", "id_right", "record_left", "record_right", "label"]
    ).reset_index(drop=True)


def sample_dataset(df: pd.DataFrame, num_pos: int, num_neg: int, seed: int) -> pd.DataFrame:
    positives = df[df["label"] == "1"].copy()
    negatives = (
        df[df["label"] == "0"]
        .sort_values("_score", ascending=False, kind="stable")
        .copy()
    )

    if len(positives) < num_pos:
        raise ValueError(f"Not enough positive samples: need {num_pos}, got {len(positives)}")
    if len(negatives) < num_neg:
        raise ValueError(f"Not enough negative samples: need {num_neg}, got {len(negatives)}")

    pos_sample = positives.sample(n=num_pos, random_state=seed)
    neg_sample = negatives.head(num_neg)
    return (
        pd.concat([pos_sample, neg_sample], ignore_index=True)
        .sample(frac=1, random_state=seed + 1)
        .reset_index(drop=True)
    )


def mask_records(df: pd.DataFrame, mask_side: str) -> pd.DataFrame:
    out = df.copy()
    if mask_side in {"left", "both"}:
        out["record_left"] = out["record_left"].map(
            lambda record: mask_col_val_field(record, RESEARCH_COL)
        )
    if mask_side in {"right", "both"}:
        out["record_right"] = out["record_right"].map(
            lambda record: mask_col_val_field(record, RESEARCH_COL)
        )
    return out


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

    candidates = prepare_samples(load_candidates(args.candidates))
    sampled = sample_dataset(candidates, args.num_pos, args.num_neg, args.seed)
    output_df = mask_records(sampled, args.mask_side)
    write_finetune_txt(output_df, args.output)

    print(f"candidate_pool={len(candidates)}")
    print(f"output_rows={len(output_df)}")
    print(f"label_counts={output_df['label'].value_counts().sort_index().to_dict()}")
    print(f"mask_side={args.mask_side}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
