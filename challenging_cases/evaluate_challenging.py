from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune import (
    SPECIAL_TOKENS,
    AGGREGATION_METHODS,
    BinaryCollator,
    ScholarBinaryClassifier,
    ScholarBinaryDataset,
    default_pretrained_name,
    evaluate,
    load_id_lookup,
    load_neighbors_json,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned SER checkpoint on a specified test file."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--dataset_A", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_A_aug.csv"))
    parser.add_argument("--dataset_B", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_B_aug.csv"))
    parser.add_argument("--dataset_A_neighbors", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_A_neighbors.json"))
    parser.add_argument("--dataset_B_neighbors", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_B_neighbors.json"))
    parser.add_argument("--neigh_id_key_A", type=str, default="a_id")
    parser.add_argument("--neigh_id_key_B", type=str, default="b_id")

    parser.add_argument("--pretrained_name", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_neighbors", type=int, default=None)
    parser.add_argument("--max_neighbor_length", type=int, default=None)
    parser.add_argument("--n_emb", type=int, default=None)
    parser.add_argument("--a_emb", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attn_type", type=str, default=None)
    parser.add_argument("--aggregation_method", type=str, choices=AGGREGATION_METHODS, default=None)
    parser.add_argument("--use_sani", type=int, choices=[0, 1], default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Use P(label=1) >= threshold instead of argmax.",
    )
    parser.add_argument(
        "--sweep_thresholds",
        action="store_true",
        help="Evaluate thresholds from 0.05 to 0.95 and report the best F1.",
    )
    return parser.parse_args()


def get_config_value(args: argparse.Namespace, ckpt_args: dict, name: str, default):
    value = getattr(args, name)
    if value is not None:
        return value
    return ckpt_args.get(name, default)


def print_neighbor_coverage(name, dataset, neighbors_A, neighbors_B) -> None:
    total = len(dataset)
    if total == 0:
        log(f"{name} neighbor coverage A/B: 0.00% / 0.00%")
        return

    covered_A = 0
    covered_B = 0
    for sample in dataset.samples:
        id_left = None if sample.get("id_left") is None else str(sample["id_left"])
        id_right = None if sample.get("id_right") is None else str(sample["id_right"])
        if neighbors_A.get(id_left):
            covered_A += 1
        if neighbors_B.get(id_right):
            covered_B += 1

    log(
        f"{name} neighbor coverage A/B: "
        f"{covered_A / total * 100:.2f}% / {covered_B / total * 100:.2f}%"
    )


def compute_neighbor_stats(dataset, neighbors_A, neighbors_B, max_neighbors: int) -> dict:
    total = len(dataset)
    stats = {
        "covered_A": 0,
        "covered_B": 0,
        "used_neighbors_A": 0,
        "used_neighbors_B": 0,
        "avg_used_neighbors_A": 0.0,
        "avg_used_neighbors_B": 0.0,
    }
    if total == 0:
        return stats

    for sample in dataset.samples:
        id_left = None if sample.get("id_left") is None else str(sample["id_left"])
        id_right = None if sample.get("id_right") is None else str(sample["id_right"])
        left_neighbors = neighbors_A.get(id_left, [])
        right_neighbors = neighbors_B.get(id_right, [])

        if left_neighbors:
            stats["covered_A"] += 1
        if right_neighbors:
            stats["covered_B"] += 1

        stats["used_neighbors_A"] += min(len(left_neighbors), max_neighbors)
        stats["used_neighbors_B"] += min(len(right_neighbors), max_neighbors)

    stats["avg_used_neighbors_A"] = stats["used_neighbors_A"] / total
    stats["avg_used_neighbors_B"] = stats["used_neighbors_B"] / total
    return stats


@torch.no_grad()
def evaluate_with_thresholds(model, dataloader, device, thresholds, desc="ThresholdEval"):
    model.eval()

    all_probs = []
    all_labels = []
    total_loss = 0.0
    total_count = 0

    for batch in __import__("tqdm").tqdm(dataloader, desc=desc, leave=False):
        from finetune import move_to_device

        batch = move_to_device(batch, device)
        loss, logits = model(**batch)
        probs = torch.softmax(logits, dim=1)[:, 1]

        bs = batch["input_ids"].size(0)
        total_loss += loss.item() * bs
        total_count += bs

        all_probs.extend(probs.detach().cpu().tolist())
        all_labels.extend(batch["labels"].detach().cpu().tolist())

    avg_loss = total_loss / max(total_count, 1)
    results = []
    for threshold in thresholds:
        preds = [1 if prob >= threshold else 0 for prob in all_probs]
        results.append(
            {
                "threshold": float(threshold),
                "loss": avg_loss,
                "acc": accuracy_score(all_labels, preds) if total_count > 0 else 0.0,
                "f1": f1_score(all_labels, preds, average="binary", pos_label=1, zero_division=0),
                "precision": precision_score(all_labels, preds, average="binary", pos_label=1, zero_division=0),
                "recall": recall_score(all_labels, preds, average="binary", pos_label=1, zero_division=0),
            }
        )

    model.train()
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    log(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if "model_state_dict" not in ckpt:
        raise ValueError(f"{args.ckpt} does not contain model_state_dict")
    log("Checkpoint loaded.")

    pretrained_name = args.pretrained_name or ckpt_args.get("pretrained_name") or default_pretrained_name()
    batch_size = int(get_config_value(args, ckpt_args, "batch_size", 16))
    max_length = int(get_config_value(args, ckpt_args, "max_length", 512))
    max_neighbors = int(get_config_value(args, ckpt_args, "max_neighbors", 5))
    max_neighbor_length = int(get_config_value(args, ckpt_args, "max_neighbor_length", 256))
    n_emb = int(get_config_value(args, ckpt_args, "n_emb", 256))
    a_emb = int(get_config_value(args, ckpt_args, "a_emb", 256))
    dropout = float(get_config_value(args, ckpt_args, "dropout", 0.1))
    attn_type = str(get_config_value(args, ckpt_args, "attn_type", "softmax"))
    aggregation_method = str(get_config_value(args, ckpt_args, "aggregation_method", "attention"))
    use_sani = bool(int(get_config_value(args, ckpt_args, "use_sani", 1)))

    log(f"Loading tokenizer from: {pretrained_name}")
    tokenizer = __import__("transformers").XLMRobertaTokenizer.from_pretrained(pretrained_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    log(f"Tokenizer loaded. vocab_size={len(tokenizer)}")

    log(f"use_sani={use_sani}")
    if use_sani:
        log(f"aggregation_method={aggregation_method}")
    left_lookup = load_id_lookup(args.dataset_A) if use_sani else {}
    right_lookup = load_id_lookup(args.dataset_B) if use_sani else {}
    neighbors_A = load_neighbors_json(args.dataset_A_neighbors, args.neigh_id_key_A) if use_sani else {}
    neighbors_B = load_neighbors_json(args.dataset_B_neighbors, args.neigh_id_key_B) if use_sani else {}
    log("ID lookup and neighbor files loaded.")

    log(f"Loading test dataset: {args.test_path}")
    dataset = ScholarBinaryDataset(
        args.test_path,
        left_lookup=left_lookup,
        right_lookup=right_lookup,
    )
    log(f"Test dataset loaded. samples={len(dataset)}")
    if use_sani:
        print_neighbor_coverage("test", dataset, neighbors_A, neighbors_B)

    log(
        "Building dataloader "
        f"(batch_size={batch_size}, max_length={max_length}, "
        f"max_neighbors={max_neighbors}, max_neighbor_length={max_neighbor_length})"
    )
    collator = BinaryCollator(
        tokenizer,
        max_length=max_length,
        max_neighbors=max_neighbors,
        max_neighbor_length=max_neighbor_length,
        neighbors_A=neighbors_A,
        neighbors_B=neighbors_B,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )
    log(f"Dataloader ready. batches={len(dataloader)}")

    log("Building model.")
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
    log("Model built. Resizing token embeddings.")
    model.language_model.resize_token_embeddings(len(tokenizer))
    log("Loading fine-tuned model_state_dict.")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    log("Model moved to device.")

    log("Starting evaluation.")
    threshold_results = None
    if args.sweep_thresholds:
        thresholds = [round(i / 100, 2) for i in range(5, 96, 5)]
        threshold_results = evaluate_with_thresholds(
            model,
            dataloader,
            device,
            thresholds=thresholds,
            desc="ThresholdSweep",
        )
        metrics = max(threshold_results, key=lambda item: item["f1"])
        log(
            "Best threshold from sweep: "
            f"{metrics['threshold']:.2f} (F1={metrics['f1']:.6f})"
        )
    elif args.threshold is not None:
        threshold_results = evaluate_with_thresholds(
            model,
            dataloader,
            device,
            thresholds=[args.threshold],
            desc="ThresholdEval",
        )
        metrics = threshold_results[0]
    else:
        metrics = evaluate(model, dataloader, device, desc="Test")
    log("Evaluation finished.")
    result = {
        "checkpoint": str(Path(args.ckpt)),
        "test_path": str(Path(args.test_path)),
        "num_samples": len(dataset),
        "use_sani": use_sani,
        "aggregation_method": aggregation_method,
        "max_neighbors": max_neighbors,
        **metrics,
    }
    if use_sani:
        result["neighbor_stats"] = compute_neighbor_stats(
            dataset,
            neighbors_A,
            neighbors_B,
            max_neighbors,
        )
    if threshold_results is not None:
        result["threshold_results"] = threshold_results

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("========== Evaluation ==========")
    print(f"Checkpoint: {result['checkpoint']}")
    print(f"Test path : {result['test_path']}")
    print(f"Samples   : {result['num_samples']}")
    if "threshold" in result:
        print(f"Threshold : {result['threshold']:.2f}")
    print(f"Precision : {result['precision']:.6f}")
    print(f"Recall    : {result['recall']:.6f}")
    print(f"F1        : {result['f1']:.6f}")
    print(f"Accuracy  : {result['acc']:.6f}")
    print(f"Loss      : {result['loss']:.6f}")
    print(f"Saved to  : {output_path}")


if __name__ == "__main__":
    main()
