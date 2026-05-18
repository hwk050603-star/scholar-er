from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/valid/test from A_B_blocking_candidates only."
    )
    parser.add_argument(
        "--candidates",
        default=str(SCRIPT_DIR / "A_B_blocking_candidates.csv"),
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--train-id-count", type=int, default=4000)
    parser.add_argument("--valid-id-count", type=int, default=1000)
    parser.add_argument("--test-id-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR))
    return parser.parse_args()


def add_sample_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_sample_key"] = (
        out["record_left"].astype(str)
        + "\t"
        + out["record_right"].astype(str)
    )
    return out


def split_complete_topk_groups(
    candidates: pd.DataFrame,
    topk: int,
    train_id_count: int,
    valid_id_count: int,
    test_id_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = (
        add_sample_key(candidates)
        .drop_duplicates("_sample_key", keep="first")
        .sort_values("_source_order", kind="stable")
        .reset_index(drop=True)
    )

    group_sizes = candidates.groupby("id_left").size()
    eligible_ids = sorted(group_sizes[group_sizes == topk].index.astype(int).tolist())

    required = train_id_count + valid_id_count + test_id_count
    if len(eligible_ids) < required:
        raise ValueError(
            "Not enough id_left groups with complete TopK candidates: "
            f"need {required}, got {len(eligible_ids)}"
        )

    selected_ids = (
        pd.Series(eligible_ids)
        .sample(n=required, random_state=seed)
        .astype(int)
        .tolist()
    )
    train_ids = set(selected_ids[:train_id_count])
    valid_ids = set(selected_ids[train_id_count : train_id_count + valid_id_count])
    test_ids = set(selected_ids[train_id_count + valid_id_count :])

    train_df = candidates[candidates["id_left"].isin(train_ids)].copy()
    valid_df = candidates[candidates["id_left"].isin(valid_ids)].copy()
    test_df = candidates[candidates["id_left"].isin(test_ids)].copy()

    return (
        train_df.sort_values("_source_order", kind="stable").reset_index(drop=True),
        valid_df.sort_values("_source_order", kind="stable").reset_index(drop=True),
        test_df.sort_values("_source_order", kind="stable").reset_index(drop=True),
    )


def write_txt(df: pd.DataFrame, path: Path) -> None:
    df[["record_left", "record_right", "label"]].to_csv(
        path,
        sep="\t",
        index=False,
        header=False,
        encoding="utf-8-sig",
    )


def shuffle_rows(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.candidates)
    candidates["_source_order"] = range(len(candidates))
    candidates["id_left"] = candidates["id_left"].astype(int)
    candidates["id_right"] = candidates["id_right"].astype(int)
    candidates["label"] = candidates["label"].astype(int)

    train_df, valid_df, test_df = split_complete_topk_groups(
        candidates=candidates,
        topk=args.topk,
        train_id_count=args.train_id_count,
        valid_id_count=args.valid_id_count,
        test_id_count=args.test_id_count,
        seed=args.seed,
    )

    train_txt = output_dir / "train.txt"
    valid_txt = output_dir / "valid.txt"
    test_txt = output_dir / "test.txt"
    test_order_txt = output_dir / "test_order.txt"

    write_txt(shuffle_rows(train_df, args.seed), train_txt)
    write_txt(shuffle_rows(valid_df, args.seed), valid_txt)
    write_txt(shuffle_rows(test_df, args.seed), test_txt)
    write_txt(test_df, test_order_txt)

    print(f"Saved: {train_txt} ({len(train_df)} rows, {train_df['id_left'].nunique()} id_left)")
    print(f"Saved: {valid_txt} ({len(valid_df)} rows, {valid_df['id_left'].nunique()} id_left)")
    print(f"Saved: {test_txt} ({len(test_df)} rows, {test_df['id_left'].nunique()} id_left)")
    print(f"Saved: {test_order_txt} ({len(test_df)} rows, {test_df['id_left'].nunique()} id_left)")
    print(f"train_labels={train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"valid_labels={valid_df['label'].value_counts().sort_index().to_dict()}")
    print(f"test_labels={test_df['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
