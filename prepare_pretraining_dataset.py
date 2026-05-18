import argparse
import random
from pathlib import Path

from pretrain_data_builders import HPCDatasetBuilder, MFPDatasetBuilder, TAMDatasetBuilder

SEED = 42
random.seed(SEED)
PROJECT_ROOT = Path(__file__).resolve().parent

def build_parser():
    parser = argparse.ArgumentParser(description="Prepare pretraining datasets for scholar entity disambiguation")
    parser.add_argument("--task", type=str, required=True, choices=["hpc", "mfp", "tam"], help="Which pretraining dataset to build")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset"))
    parser.add_argument("--max_samples", type=int, default=None, help="仅处理前 N 条样本，用于调试")

    parser.add_argument("--dataset_A", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_A_aug.csv"))
    parser.add_argument("--dataset_B", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_B_aug.csv"))
    parser.add_argument("--dataset_A_neighbors", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_A_neighbors.json"))
    parser.add_argument("--dataset_B_neighbors", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_B_neighbors.json"))
    parser.add_argument("--mapping_csv", type=str, default=str(PROJECT_ROOT / "blocking" / "A_B_mapping.csv"))
    parser.add_argument("--same_name_negatives_csv", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_aug_same_name.csv"))
    parser.add_argument("--blocking_candidates_csv", type=str, default=str(PROJECT_ROOT / "blocking" / "A_B_blocking_candidates.csv"))
    parser.add_argument(
        "--partial_work_history_csv",
        type=str,
        default=str(PROJECT_ROOT / "pretrain" / "dataset" / "partial_work_history.csv"),
    )
    parser.add_argument("--input_csv", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_A_aug.csv"))
    parser.add_argument("--input_neighbors_json", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_A_neighbors.json"))

    parser.add_argument("--id_col_A", type=str, default="Id")
    parser.add_argument("--id_col_B", type=str, default="Id")
    parser.add_argument("--id_col", type=str, default="Id")
    parser.add_argument("--research_col", type=str, default="Research Interests")
    parser.add_argument("--neigh_id_key_A", type=str, default="a_id")
    parser.add_argument("--neigh_id_key_B", type=str, default="b_id")

    parser.add_argument("--max_neighbors", type=int, default=5)

    parser.add_argument("--use_pinyin_aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_easy_neg_per_pos", type=int, default=1)
    parser.add_argument("--num_hard_neg_per_pos", type=int, default=4)
    parser.add_argument("--compact_hpc_json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop_id_from_hpc_attr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--omit_hpc_neighbors", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--mask_token", type=str, default="<mask>")
    parser.add_argument("--max_local_fields", type=int, default=10)
    parser.add_argument("--mfp_mask_ratio", type=float, default=0.3)
    parser.add_argument("--mfp_mask_prob", type=float, default=0.5)

    parser.add_argument("--num_pos_per_entity", type=int, default=1)
    parser.add_argument("--num_same_name_neg_per_entity", type=int, default=1)
    parser.add_argument("--num_easy_neg_per_entity", type=int, default=1)
    parser.add_argument("--num_hard_neg_per_entity", type=int, default=1)
    parser.add_argument("--exclude_org_keyword", type=str, default="计算技术研究所")

    return parser


def validate_args(args):
    if args.task == "hpc":
        required = [
            "dataset_A", "dataset_B", "dataset_A_neighbors",
            "dataset_B_neighbors", "mapping_csv", "output_dir"
        ]
    elif args.task == "mfp":
        required = ["input_csv", "input_neighbors_json", "output_dir"]
    elif args.task == "tam":
        required = [
            "dataset_A", "dataset_B", "dataset_A_neighbors",
            "dataset_B_neighbors", "partial_work_history_csv", "output_dir"
        ]
    else:
        raise ValueError(f"Unknown task: {args.task}")

    missing = [k for k in required if getattr(args, k) in [None, ""]]
    if missing:
        raise ValueError(f"Missing required arguments for task={args.task}: {missing}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    builder_map = {
        "hpc": HPCDatasetBuilder,
        "mfp": MFPDatasetBuilder,
        "tam": TAMDatasetBuilder,
    }

    builder = builder_map[args.task](args)
    builder.run()
