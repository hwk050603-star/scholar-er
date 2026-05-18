from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from transformers import XLMRobertaTokenizer

CASE_STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CASE_STUDY_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune import (
    AGGREGATION_METHODS,
    SPECIAL_TOKENS,
    BinaryCollator,
    ScholarBinaryClassifier,
    default_pretrained_name,
    extract_attr_field,
    find_entity_id,
    load_id_lookup,
    load_neighbors_json,
    move_to_device,
)


FIELDNAMES = [
    "left_id",
    "right_id",
    "left_source",
    "right_source",
    "left_name",
    "right_name",
    "label",
    "prediction",
    "match_probability",
    "side",
    "entity_id",
    "entity_name",
    "neighbor_rank",
    "beta",
    "neighbor_text",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fine-tuned SANI model on case-study pairs and export "
            "neighbor relevance scores plus softmax beta weights."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(PROJECT_ROOT / "finetune_checkpoints" / "model.pth"),
        help="Fine-tuned checkpoint containing model_state_dict.",
    )
    parser.add_argument(
        "--case_path",
        type=str,
        default=str(CASE_STUDY_DIR / "case_study_cases.tsv"),
        help=(
            "TSV case file. Preferred format: id_left, id_right, left_text, "
            "right_text, optional label. The script uses IDs directly to fetch "
            "top-K neighbors."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CASE_STUDY_DIR / "case_neighbor_weights.csv"),
        help="CSV path for exported case-study neighbor weights.",
    )
    parser.add_argument("--dataset_A", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_A_aug.csv"))
    parser.add_argument("--dataset_B", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_B_aug.csv"))
    parser.add_argument(
        "--dataset_A_neighbors",
        type=str,
        default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_A_neighbors.json"),
    )
    parser.add_argument(
        "--dataset_B_neighbors",
        type=str,
        default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_B_neighbors.json"),
    )
    parser.add_argument("--neigh_id_key_A", type=str, default="a_id")
    parser.add_argument("--neigh_id_key_B", type=str, default="b_id")
    parser.add_argument(
        "--left_source",
        type=str,
        choices=["A", "B"],
        default="A",
        help="Source used to look up left-side neighbors.",
    )
    parser.add_argument(
        "--right_source",
        type=str,
        choices=["A", "B"],
        default="B",
        help="Source used to look up right-side neighbors. Use A for A-A case studies.",
    )
    parser.add_argument("--pretrained_name", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_neighbors", type=int, default=None)
    parser.add_argument("--max_neighbor_length", type=int, default=None)
    parser.add_argument("--n_emb", type=int, default=None)
    parser.add_argument("--a_emb", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attn_type", type=str, default=None)
    parser.add_argument("--aggregation_method", type=str, choices=AGGREGATION_METHODS, default=None)
    parser.add_argument("--use_sani", type=int, choices=[0, 1], default=None)
    return parser.parse_args()


def get_config_value(args: argparse.Namespace, ckpt_args: dict, name: str, default):
    value = getattr(args, name)
    if value is not None:
        return value
    return ckpt_args.get(name, default)


def load_checkpoint(path: str, device: torch.device) -> tuple[dict, dict]:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"{ckpt_path} does not contain model_state_dict")
    return ckpt, ckpt.get("args", {})


def build_model_and_tokenizer(args: argparse.Namespace, ckpt: dict, ckpt_args: dict, device: torch.device):
    pretrained_name = args.pretrained_name or ckpt_args.get("pretrained_name") or default_pretrained_name()
    max_length = int(get_config_value(args, ckpt_args, "max_length", 512))
    max_neighbors = int(get_config_value(args, ckpt_args, "max_neighbors", 5))
    max_neighbor_length = int(get_config_value(args, ckpt_args, "max_neighbor_length", 256))
    n_emb = int(get_config_value(args, ckpt_args, "n_emb", 256))
    a_emb = int(get_config_value(args, ckpt_args, "a_emb", 256))
    dropout = float(get_config_value(args, ckpt_args, "dropout", 0.1))
    attn_type = str(get_config_value(args, ckpt_args, "attn_type", "softmax"))
    aggregation_method = str(get_config_value(args, ckpt_args, "aggregation_method", "attention"))
    use_sani = bool(int(get_config_value(args, ckpt_args, "use_sani", 1)))

    if not use_sani:
        raise ValueError("The checkpoint was configured with use_sani=0, so beta weights are unavailable.")
    if attn_type != "softmax":
        raise ValueError(f"SANI beta export expects attn_type=softmax, got {attn_type}.")

    log(f"Loading tokenizer from: {pretrained_name}")
    tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    log("Building fine-tuned SANI model.")
    model = ScholarBinaryClassifier(
        device=device,
        pretrained_name=pretrained_name,
        dropout=dropout,
        num_labels=2,
        n_emb=n_emb,
        a_emb=a_emb,
        attn_type=attn_type,
        use_sani=use_sani,
        aggregation_method=aggregation_method,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    config = {
        "pretrained_name": pretrained_name,
        "max_length": max_length,
        "max_neighbors": max_neighbors,
        "max_neighbor_length": max_neighbor_length,
        "n_emb": n_emb,
        "a_emb": a_emb,
        "dropout": dropout,
        "attn_type": attn_type,
        "aggregation_method": aggregation_method,
        "use_sani": use_sani,
    }
    return model, tokenizer, config


def parse_case_file(path: str, left_lookup: dict | None = None, right_lookup: dict | None = None) -> list[dict]:
    cases = []
    left_lookup = left_lookup or {}
    right_lookup = right_lookup or {}

    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue

            parts = [part.strip() for part in line.split("\t")]
            if len(parts) == 5:
                id_left, id_right, left_text, right_text, label_str = parts
                label = int(label_str)
            elif len(parts) == 4:
                id_left, id_right, left_text, right_text = parts
                label = 0
            elif len(parts) == 3:
                left_text, right_text, label_str = parts
                id_left = find_entity_id(left_text, left_lookup)
                id_right = find_entity_id(right_text, right_lookup)
                label = int(label_str)
            else:
                raise ValueError(
                    f"[Line {line_id}] Expected 4/5 TSV columns with IDs "
                    "or 3 columns without IDs. "
                    f"Got {len(parts)} columns."
                )

            cases.append(
                {
                    "left_text": left_text,
                    "right_text": right_text,
                    "id_left": None if id_left == "" else str(id_left),
                    "id_right": None if id_right == "" else str(id_right),
                    "label": label,
                }
            )

    return cases


def load_cases(args: argparse.Namespace, use_sani: bool) -> tuple[list[dict], dict[str, dict]]:
    left_lookup = load_id_lookup(args.dataset_A) if use_sani else {}
    right_lookup = load_id_lookup(args.dataset_B) if use_sani else {}
    neighbors_A = load_neighbors_json(args.dataset_A_neighbors, args.neigh_id_key_A) if use_sani else {}
    neighbors_B = load_neighbors_json(args.dataset_B_neighbors, args.neigh_id_key_B) if use_sani else {}

    cases = parse_case_file(args.case_path, left_lookup=left_lookup, right_lookup=right_lookup)
    if len(cases) == 0:
        raise ValueError(f"No cases loaded from {args.case_path}")
    return cases, {"A": neighbors_A, "B": neighbors_B}


def neighbor_scores_for_side(model, center_fea, neigh_ids_list, neigh_mask_list):
    responses = []
    for token_n, mask_n in zip(neigh_ids_list, neigh_mask_list):
        response = model.samp_layer._encode_single_neighbor_response(
            center_fea=center_fea,
            token_n=token_n,
            mask_n=mask_n,
            neighbert=model.neighbert,
        )
        responses.append(response)

    if not responses:
        return [], []

    u_list = [z.mean(dim=1) for z in responses]
    U = torch.cat(u_list, dim=0)
    relevance = model.samp_layer.w(torch.tanh(model.samp_layer.W_u(U))).view(-1)
    beta = torch.softmax(relevance, dim=0)
    return relevance.detach().cpu().tolist(), beta.detach().cpu().tolist()


@torch.no_grad()
def infer_one_case(
    model,
    collator,
    sample: dict,
    sample_id: int,
    left_neighbor_map: dict,
    right_neighbor_map: dict,
    left_source: str,
    right_source: str,
):
    batch = collator([sample])
    batch = move_to_device(batch, model.device)

    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "x_n": batch["x_n"],
        "entity_pos_list": batch["entity_pos_list"],
    }
    logits = model(**model_inputs)
    probabilities = torch.softmax(logits, dim=1)[0]
    prediction = int(torch.argmax(logits, dim=1).item())
    match_probability = float(probabilities[1].detach().cpu().item())

    encoder_output = model.language_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).last_hidden_state

    (e1_start, e1_end), (e2_start, e2_end) = batch["entity_pos_list"][0]
    left_center = encoder_output[0, e1_start:e1_end, :].unsqueeze(0)
    right_center = encoder_output[0, e2_start:e2_end, :].unsqueeze(0)
    x_n = batch["x_n"][0]

    left_scores, left_betas = neighbor_scores_for_side(
        model=model,
        center_fea=left_center,
        neigh_ids_list=x_n.get("neigh1_input_ids", []),
        neigh_mask_list=x_n.get("neigh1_attention_mask", []),
    )
    right_scores, right_betas = neighbor_scores_for_side(
        model=model,
        center_fea=right_center,
        neigh_ids_list=x_n.get("neigh2_input_ids", []),
        neigh_mask_list=x_n.get("neigh2_attention_mask", []),
    )

    left_id = "" if sample.get("id_left") is None else str(sample["id_left"])
    right_id = "" if sample.get("id_right") is None else str(sample["id_right"])
    left_neighbors = list(left_neighbor_map.get(left_id, [])[: collator.max_neighbors])
    right_neighbors = list(right_neighbor_map.get(right_id, [])[: collator.max_neighbors])

    left_name = extract_attr_field(sample["left_text"], "Name")
    right_name = extract_attr_field(sample["right_text"], "Name")
    common = {
        "left_id": left_id,
        "right_id": right_id,
        "left_source": left_source,
        "right_source": right_source,
        "left_name": left_name,
        "right_name": right_name,
        "label": sample["label"],
        "prediction": prediction,
        "match_probability": match_probability,
    }

    rows = []
    for side, entity_id, entity_name, scores, betas, neighbor_texts in (
        ("left", left_id, left_name, left_scores, left_betas, left_neighbors),
        ("right", right_id, right_name, right_scores, right_betas, right_neighbors),
    ):
        for rank, (score, beta) in enumerate(zip(scores, betas), start=1):
            neighbor_text = neighbor_texts[rank - 1] if rank <= len(neighbor_texts) else ""
            rows.append(
                {
                    **common,
                    "side": side,
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "neighbor_rank": rank,
                    "beta": beta,
                    "neighbor_text": neighbor_text,
                }
            )
    return rows


def write_rows(rows: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    log(f"Saved neighbor weights to: {path}")


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    ckpt, ckpt_args = load_checkpoint(args.ckpt, device)
    model, tokenizer, config = build_model_and_tokenizer(args, ckpt, ckpt_args, device)
    cases, neighbor_maps = load_cases(args, config["use_sani"])
    left_neighbor_map = neighbor_maps[args.left_source]
    right_neighbor_map = neighbor_maps[args.right_source]

    collator = BinaryCollator(
        tokenizer,
        max_length=config["max_length"],
        max_neighbors=config["max_neighbors"],
        max_neighbor_length=config["max_neighbor_length"],
        neighbors_A=left_neighbor_map,
        neighbors_B=right_neighbor_map,
    )

    all_rows = []
    for sample_id, sample in enumerate(cases):
        all_rows.extend(
            infer_one_case(
                model=model,
                collator=collator,
                sample=sample,
                sample_id=sample_id,
                left_neighbor_map=left_neighbor_map,
                right_neighbor_map=right_neighbor_map,
                left_source=args.left_source,
                right_source=args.right_source,
            )
        )

    metadata_path = Path(args.output).with_suffix(".meta.json")
    metadata = {
        "checkpoint": str(Path(args.ckpt)),
        "case_path": str(Path(args.case_path)),
        "output": str(Path(args.output)),
        "num_cases": len(cases),
        "num_rows": len(all_rows),
        "left_source": args.left_source,
        "right_source": args.right_source,
        "config": config,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_rows(all_rows, args.output)
    log(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
