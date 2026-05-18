import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import AGGREGATION_METHODS, ScholarSampForPretrain


SPECIAL_TOKENS = ["COL", "VAL"]


def default_pretrained_name():
    local_model_dir = PROJECT_ROOT / "xlm-roberta-base"
    if local_model_dir.exists():
        return str(local_model_dir)
    return "xlm-roberta-base"


def load_neighbors_json(path, id_key):
    if not path or not os.path.exists(path):
        print(f"Warning: neighbor file not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    neigh_map = {}
    for item in data:
        idx = str(item[id_key])
        neigh_map[idx] = item.get("neighs_attr", [])
    return neigh_map


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dict_to_device(d, device):
    if isinstance(d, dict):
        return {k: dict_to_device(v, device) for k, v in d.items()}
    elif isinstance(d, list):
        return [dict_to_device(v, device) for v in d]
    elif torch.is_tensor(d):
        return d.to(device)
    else:
        return d


class MultiTaskDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=None):
        self.samples = []
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} does not exist. Returning empty dataset.")
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return

        decoder = json.JSONDecoder()
        idx, length = 0, len(content)
        while idx < length:
            while idx < length and content[idx].isspace():
                idx += 1
            if idx >= length:
                break

            s, idx = decoder.raw_decode(content, idx)
            self.samples.append(s)

            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def split_dataset(dataset, train_ratio=0.9, seed=42):
    if len(dataset) == 0:
        return dataset, dataset

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    n_train = int(len(indices) * train_ratio)

    train_ds = MultiTaskDataset.__new__(MultiTaskDataset)
    dev_ds = MultiTaskDataset.__new__(MultiTaskDataset)
    train_ds.samples = [dataset.samples[i] for i in indices[:n_train]]
    dev_ds.samples = [dataset.samples[i] for i in indices[n_train:]]
    return train_ds, dev_ds


def compute_tam_class_weights(dataset, manual_pos_weight=0.0):
    labels = [int(s.get("label", 0)) for s in getattr(dataset, "samples", [])]
    num_pos = sum(1 for y in labels if y == 1)
    num_neg = sum(1 for y in labels if y == 0)

    if manual_pos_weight and manual_pos_weight > 0:
        weights = [1.0, float(manual_pos_weight)]
    elif num_pos > 0 and num_neg > 0:
        total = num_pos + num_neg
        weights = [
            total / (2.0 * num_neg),
            total / (2.0 * num_pos)
        ]
    else:
        weights = [1.0, 1.0]

    return weights, num_neg, num_pos


def compute_mfp_pos_weight(dataset, num_fields, max_pos_weight=5.0):
    pos_counts = [0.0] * num_fields
    total = 0

    for sample in getattr(dataset, "samples", []):
        labels = list(sample.get("target_multi_hot", []))[:num_fields]
        if len(labels) < num_fields:
            labels += [0.0] * (num_fields - len(labels))

        total += 1
        for i, value in enumerate(labels):
            if float(value) > 0:
                pos_counts[i] += 1.0

    if total == 0:
        return [1.0] * num_fields, pos_counts, [0.0] * num_fields

    weights = []
    pos_rates = []
    for pos in pos_counts:
        neg = total - pos
        pos_rates.append(pos / total)
        if pos <= 0 or neg <= 0:
            weights.append(1.0)
        else:
            weights.append(min(max_pos_weight, neg / pos))

    return weights, pos_counts, pos_rates


class TaskCollator:
    def __init__(self, tokenizer, args, hpc_neighbor_maps=None):
        self.tokenizer = tokenizer
        self.args = args
        self.hpc_neighbor_maps = hpc_neighbor_maps or {"A": {}, "B": {}}

    def _build_pair_input(self, text1, text2):
        t1_ids = self.tokenizer.encode(text1, add_special_tokens=False)
        t2_ids = self.tokenizer.encode(text2, add_special_tokens=False)
        max_t = self.args.max_length - 4

        if len(t1_ids) + len(t2_ids) > max_t:
            half = max_t // 2
            if len(t1_ids) > half and len(t2_ids) > half:
                t1_ids, t2_ids = t1_ids[:half], t2_ids[:(max_t - half)]
            elif len(t1_ids) > half:
                t1_ids = t1_ids[:(max_t - len(t2_ids))]
            else:
                t2_ids = t2_ids[:(max_t - len(t1_ids))]

        input_ids = (
            [self.tokenizer.bos_token_id]
            + t1_ids
            + [self.tokenizer.eos_token_id] * 2
            + t2_ids
            + [self.tokenizer.eos_token_id]
        )
        e1_pos = (1, 1 + len(t1_ids))
        e2_pos = (e1_pos[1] + 2, e1_pos[1] + 2 + len(t2_ids))

        mask = [1] * len(input_ids)
        pad_len = self.args.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        mask += [0] * pad_len
        return input_ids, mask, (e1_pos, e2_pos)

    def _build_single_input(self, text):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = token_ids[:self.args.max_length - 2]
        input_ids = (
            [self.tokenizer.bos_token_id]
            + token_ids
            + [self.tokenizer.eos_token_id]
        )
        entity_pos = (1, 1 + len(token_ids))
        mask = [1] * len(input_ids)
        pad_len = self.args.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        mask += [0] * pad_len
        return input_ids, mask, entity_pos

    def _append_tokens(self, target, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        target.extend(tokens)
        return len(tokens)

    def _build_mfp_input(self, sample):
        input_text = sample.get("input_text", "")
        corrupted_fields = sample.get("corrupted_research_fields")
        if corrupted_fields is None:
            corrupted_fields = [self.args.mask_token]
        selected_indices = set(int(i) for i in sample.get("masked_field_indices", []))

        token_ids = [self.tokenizer.bos_token_id]
        field_spans = []

        marker = "COL Research Interests VAL "
        marker_pos = input_text.find(marker)
        cursor = marker_pos + len(marker) if marker_pos >= 0 else 0
        if cursor > 0:
            self._append_tokens(token_ids, input_text[:cursor])

        for idx, field in enumerate(corrupted_fields):
            field = str(field)
            field_start = input_text.find(field, cursor)
            if field_start < 0:
                continue

            self._append_tokens(token_ids, input_text[cursor:field_start])
            start = len(token_ids)
            self._append_tokens(token_ids, field)
            end = len(token_ids)
            if idx in selected_indices:
                field_spans.append((start, end))
            cursor = field_start + len(field)

        self._append_tokens(token_ids, input_text[cursor:])
        token_ids.append(self.tokenizer.eos_token_id)

        if len(token_ids) > self.args.max_length:
            token_ids = token_ids[:self.args.max_length]
            field_spans = [
                (start, min(end, self.args.max_length))
                for start, end in field_spans
                if start < self.args.max_length
            ]

        valid_len = len(token_ids)
        mask = [1] * valid_len
        pad_len = self.args.max_length - valid_len
        token_ids += [self.tokenizer.pad_token_id] * pad_len
        mask += [0] * pad_len

        if not field_spans:
            field_spans = [(0, 1)]
        entity_pos = (1, max(valid_len - 1, 1))
        return token_ids, mask, entity_pos, field_spans

    def _encode_neighbors(self, neighbors):
        if not bool(self.args.use_sani):
            return [], []

        ids, masks = [], []
        for text in (neighbors or [])[:self.args.max_neighbors]:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.args.max_neighbor_length,
                padding="max_length"
            )
            ids.append(enc["input_ids"])
            masks.append(enc["attention_mask"])
        return ids, masks

    def _get_entity_neighbors(self, item, default_source):
        if "neighbors_attr" in item:
            return item.get("neighbors_attr", [])
        source = item.get("source", default_source)
        entity_id = str(item.get("id", item.get("sample_id", "")))
        return self.hpc_neighbor_maps.get(source, {}).get(entity_id, [])

    def collate_hpc(self, batch):
        ids, atts, pos, x_n, labels = [], [], [], [], []
        for s in batch:
            if "left" not in s or "right" not in s:
                raise ValueError(
                    "HPC training requires pair samples with 'left', 'right', and 'label'. "
                    "Please regenerate hpc_dataset.jsonl."
                )

            i, m, p = self._build_pair_input(s["left"]["attr"], s["right"]["attr"])
            n1_i, n1_m = self._encode_neighbors(self._get_entity_neighbors(s["left"], "A"))
            n2_i, n2_m = self._encode_neighbors(self._get_entity_neighbors(s["right"], "B"))
            ids.append(i)
            atts.append(m)
            pos.append(p)
            labels.append(s["label"])
            x_n.append({
                "neigh1_input_ids": torch.tensor(n1_i, dtype=torch.long),
                "neigh1_attention_mask": torch.tensor(n1_m, dtype=torch.long),
                "neigh2_input_ids": torch.tensor(n2_i, dtype=torch.long),
                "neigh2_attention_mask": torch.tensor(n2_m, dtype=torch.long)
            })

        return {
            "x": torch.tensor(ids),
            "att_mask": torch.tensor(atts),
            "entity_pos_list": pos,
            "x_n": x_n,
            "pair_labels": torch.tensor(labels)
        }

    def collate_tam(self, batch):
        ids, atts, pos, x_n, labels = [], [], [], [], []
        for s in batch:
            i, m, p = self._build_pair_input(s["entity_attr"], s["candidate_affiliation_attr"])
            n1_i, n1_m = self._encode_neighbors(self._get_entity_neighbors(s, "A"))
            ids.append(i)
            atts.append(m)
            pos.append(p)
            labels.append(s["label"])
            x_n.append({
                "neigh1_input_ids": torch.tensor(n1_i, dtype=torch.long),
                "neigh1_attention_mask": torch.tensor(n1_m, dtype=torch.long),
                "neigh2_input_ids": [],
                "neigh2_attention_mask": []
            })
        return {
            "x": torch.tensor(ids),
            "att_mask": torch.tensor(atts),
            "entity_pos_list": pos,
            "x_n": x_n,
            "pair_labels": torch.tensor(labels)
        }

    def collate_mfp(self, batch):
        ids, atts, pos, x_n, labels, spans = [], [], [], [], [], []
        field_ids, field_atts = [], []
        for s in batch:
            input_ids, attention_mask, entity_pos, field_spans = self._build_mfp_input(s)
            n_i, n_m = self._encode_neighbors(self._get_entity_neighbors(s, "A"))
            ids.append(input_ids)
            atts.append(attention_mask)
            pos.append(entity_pos)
            spans.append(field_spans)
            x_n.append({
                "neighbors_input_ids": torch.tensor(n_i, dtype=torch.long),
                "neighbors_attention_mask": torch.tensor(n_m, dtype=torch.long)
            })

            candidate_fields = s.get("candidate_fields", [])[:self.args.num_fields]
            if len(candidate_fields) < self.args.num_fields:
                candidate_fields = candidate_fields + [""] * (self.args.num_fields - len(candidate_fields))
            field_enc = self.tokenizer(
                candidate_fields,
                truncation=True,
                max_length=self.args.max_field_length,
                padding="max_length"
            )
            field_ids.append(field_enc["input_ids"])
            field_atts.append(field_enc["attention_mask"])

            l = s["target_multi_hot"][:self.args.num_fields]
            if len(l) < self.args.num_fields:
                l += [0.0] * (self.args.num_fields - len(l))
            labels.append(l)

        return {
            "x": torch.tensor(ids),
            "att_mask": torch.tensor(atts),
            "entity_pos_list": pos,
            "x_n": x_n,
            "mfp_labels": torch.tensor(labels),
            "mfp_field_spans": spans,
            "field_input_ids": torch.tensor(field_ids),
            "field_attention_mask": torch.tensor(field_atts)
        }


@torch.no_grad()
def evaluate_task(model, dataloader, task_type, device, args=None):
    if len(dataloader) == 0:
        return {"loss": 0.0, "f1": 0.0, "acc": 0.0}

    model.eval()
    total_loss, total_samples = 0.0, 0

    tp = fp = fn = tn = 0
    mfp_correct, mfp_total_preds, mfp_total_trues = 0, 0, 0

    for batch in tqdm(dataloader, desc=f"Eval {task_type.upper()}", leave=False):
        batch = dict_to_device(batch, device)
        loss, logits = model(task_type=task_type, **batch)
        bsz = batch["x"].size(0)
        total_loss += loss.item() * bsz
        total_samples += bsz

        if task_type == "hpc":
            preds = torch.argmax(logits, dim=1)
            labels = batch["pair_labels"]
            for p, y in zip(preds.tolist(), labels.tolist()):
                if p == 1 and y == 1:
                    tp += 1
                elif p == 1 and y == 0:
                    fp += 1
                elif p == 0 and y == 1:
                    fn += 1
                elif p == 0 and y == 0:
                    tn += 1
        elif task_type == "tam":
            preds = torch.argmax(logits, dim=1)
            labels = batch["pair_labels"]
            for p, y in zip(preds.tolist(), labels.tolist()):
                if p == 1 and y == 1:
                    tp += 1
                elif p == 1 and y == 0:
                    fp += 1
                elif p == 0 and y == 1:
                    fn += 1
                elif p == 0 and y == 0:
                    tn += 1
        elif task_type == "mfp":
            probs = torch.sigmoid(logits)
            threshold = 0.5 if args is None else args.mfp_threshold
            preds = (probs > threshold).float()
            labels = batch["mfp_labels"]
            mfp_correct += (preds == labels).all(dim=1).sum().item()
            mfp_total_preds += preds.sum().item()
            mfp_total_trues += labels.sum().item()
            tp += (preds * labels).sum().item()

    avg_loss = total_loss / max(total_samples, 1)

    if task_type == "hpc":
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        pred_pos = tp + fp
        pred_neg = tn + fn
        label_pos = tp + fn
        label_neg = tn + fp
    elif task_type == "tam":
        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        pred_pos = tp + fp
        pred_neg = tn + fn
        label_pos = tp + fn
        label_neg = tn + fp
    else:
        acc = mfp_correct / max(total_samples, 1)
        precision = tp / max(mfp_total_preds, 1)
        recall = tp / max(mfp_total_trues, 1)

    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    model.train()
    result = {"loss": avg_loss, "acc": acc, "f1": f1}
    if task_type in {"hpc", "tam"}:
        result.update(
            {
                "precision": precision,
                "recall": recall,
                "pred_pos": pred_pos,
                "pred_neg": pred_neg,
                "label_pos": label_pos,
                "label_neg": label_neg,
            }
        )
    if task_type == "mfp":
        result.update(
            {
                "precision": precision,
                "recall": recall,
                "pred_pos": mfp_total_preds,
                "label_pos": mfp_total_trues,
            }
        )
    return result


def _run_evaluation(model, dev_loaders, device, epoch, optimizer, scheduler, best_global_score, model_path, args):
    metrics = {}
    global_score = 0.0
    total_weight = 0.0

    for task_type, loader in dev_loaders.items():
        if len(loader) > 0:
            res = evaluate_task(model, loader, task_type, device, args=args)
            metrics[task_type] = res
            detail = ""
            if task_type in {"hpc", "tam"}:
                detail = (
                    f" | P: {res['precision']:.4f} | R: {res['recall']:.4f}"
                    f" | Pred+/Label+: {res['pred_pos']}/{res['label_pos']}"
                )
            elif task_type == "mfp":
                detail = (
                    f" | P: {res['precision']:.4f} | R: {res['recall']:.4f}"
                    f" | Pred+/Label+: {res['pred_pos']:.0f}/{res['label_pos']:.0f}"
                )
            print(f"[{task_type.upper()}] Loss: {res['loss']:.4f} | F1: {res['f1']:.4f} | Acc: {res['acc']:.4f}{detail}")
            task_weight = float(getattr(args, f"w_{task_type}", 1.0))
            global_score += task_weight * res["f1"]
            total_weight += task_weight

    if total_weight > 0:
        global_score = global_score / total_weight
        print(f"==> Weighted Global F1: {global_score:.4f}")

    if global_score > best_global_score:
        best_global_score = global_score
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_global_score": best_global_score,
            "metrics": metrics,
            "args": vars(args)
        }
        torch.save(ckpt, model_path)
        print(f"--> New Best Multi-task Model Saved to {model_path}!\n")

    return best_global_score


def main(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    weights = {"hpc": args.w_hpc, "mfp": args.w_mfp, "tam": args.w_tam}
    active_tasks = {task for task, weight in weights.items() if weight > 0}
    print(
        "Active pre-training tasks: "
        + (", ".join(sorted(active_tasks)) if active_tasks else "none")
    )
    print(f"Use SANI module: {bool(args.use_sani)}")
    if args.use_sani:
        print(f"Neighbor aggregation method: {args.aggregation_method}")
    if not active_tasks:
        print("No active pre-training tasks because all task weights are <= 0. Exiting.")
        return

    # Load tokenizer and add COL / VAL as special tokens.
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_name)
    num_added_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS}
    )
    print(f"Added schema tokens: {SPECIAL_TOKENS}")
    print(f"Number of newly added tokens: {num_added_tokens}")
    print(f"Tokenizer vocab size after extension: {len(tokenizer)}")

    print("Loading datasets...")
    ds_hpc = MultiTaskDataset(args.hpc_jsonl)
    ds_mfp = MultiTaskDataset(args.mfp_jsonl)
    ds_tam = MultiTaskDataset(args.tam_jsonl)
    if bool(args.use_sani):
        hpc_neighbor_maps = {
            "A": load_neighbors_json(args.dataset_A_neighbors, args.neigh_id_key_A),
            "B": load_neighbors_json(args.dataset_B_neighbors, args.neigh_id_key_B)
        }
    else:
        hpc_neighbor_maps = {"A": {}, "B": {}}
        print("Use SANI is disabled; skip loading neighbor JSON files.")

    train_hpc, dev_hpc = split_dataset(ds_hpc, args.train_ratio, args.seed)
    train_mfp, dev_mfp = split_dataset(ds_mfp, args.train_ratio, args.seed)
    train_tam, dev_tam = split_dataset(ds_tam, args.train_ratio, args.seed)
    tam_class_weights, tam_train_neg, tam_train_pos = compute_tam_class_weights(
        train_tam,
        manual_pos_weight=args.tam_pos_weight
    )
    mfp_pos_weight, mfp_pos_counts, mfp_pos_rates = compute_mfp_pos_weight(
        train_mfp,
        num_fields=args.num_fields,
        max_pos_weight=args.mfp_max_pos_weight
    )

    print(f"Train sizes -> HPC: {len(train_hpc)}, MFP: {len(train_mfp)}, TAM: {len(train_tam)}")

    collator = TaskCollator(tokenizer, args, hpc_neighbor_maps=hpc_neighbor_maps)
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers, "shuffle": True}

    loader_hpc = DataLoader(train_hpc, collate_fn=collator.collate_hpc, **loader_args) if "hpc" in active_tasks and len(train_hpc) else []
    loader_mfp = DataLoader(train_mfp, collate_fn=collator.collate_mfp, **loader_args) if "mfp" in active_tasks and len(train_mfp) else []
    loader_tam = DataLoader(train_tam, collate_fn=collator.collate_tam, **loader_args) if "tam" in active_tasks and len(train_tam) else []

    eval_args = {"batch_size": args.batch_size, "shuffle": False}
    dev_loaders = {
        "hpc": DataLoader(dev_hpc, collate_fn=collator.collate_hpc, **eval_args) if "hpc" in active_tasks and len(dev_hpc) else [],
        "mfp": DataLoader(dev_mfp, collate_fn=collator.collate_mfp, **eval_args) if "mfp" in active_tasks and len(dev_mfp) else [],
        "tam": DataLoader(dev_tam, collate_fn=collator.collate_tam, **eval_args) if "tam" in active_tasks and len(dev_tam) else []
    }

    model = ScholarSampForPretrain(
        device=device,
        num_fields=args.num_fields,
        n_emb=args.n_emb,
        a_emb=args.a_emb,
        dropout=args.dropout,
        attn_type=args.attn_type,
        pretrained_name=args.pretrained_name,
        tam_class_weights=tam_class_weights,
        mfp_pos_weight=mfp_pos_weight,
        use_sani=bool(args.use_sani),
        aggregation_method=args.aggregation_method
    ).to(device)

    # Resize embedding matrix so COL / VAL obtain newly initialized trainable embeddings.
    model.language_model.resize_token_embeddings(len(tokenizer))

    # Keep tokenizer consistent between collator and model.
    model.tokenizer = tokenizer

    print(f"Model embedding matrix resized to vocab size: {len(tokenizer)}")
    print("The embeddings of COL and VAL are newly initialized and will be optimized during SER-oriented pre-training.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_batches_per_epoch = len(loader_hpc) + len(loader_mfp) + len(loader_tam)
    total_steps = total_batches_per_epoch * args.max_epochs
    if total_steps == 0:
        print("All datasets are empty. Exiting.")
        return

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    save_dir = Path(args.save_dir)
    default_save_dir = PROJECT_ROOT / "checkpoints"
    if not bool(args.use_sani) and save_dir.resolve() == default_save_dir.resolve():
        save_dir = PROJECT_ROOT / "checkpoints_no_sani"
        print(f"Use SANI is disabled; saving checkpoint to {save_dir} to avoid overwriting SANI pre-training weights.")
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pth"

    best_global_score = -1.0

    for epoch in range(1, args.max_epochs + 1):
        model.train()

        task_choices = []
        if "hpc" in active_tasks:
            task_choices += ["hpc"] * len(loader_hpc)
        if "mfp" in active_tasks:
            task_choices += ["mfp"] * len(loader_mfp)
        if "tam" in active_tasks:
            task_choices += ["tam"] * len(loader_tam)
        random.shuffle(task_choices)

        iterators = {
            "hpc": iter(loader_hpc) if len(loader_hpc) else None,
            "mfp": iter(loader_mfp) if len(loader_mfp) else None,
            "tam": iter(loader_tam) if len(loader_tam) else None
        }

        pbar = tqdm(task_choices, desc=f"Epoch {epoch}")
        logs = {"L_hpc": 0.0, "L_mfp": 0.0, "L_tam": 0.0}

        for step, task_type in enumerate(pbar):
            optimizer.zero_grad()

            batch = next(iterators[task_type])
            batch = dict_to_device(batch, device)

            loss, _ = model(task_type=task_type, **batch)
            weighted_loss = loss * weights[task_type]
            weighted_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            logs[f"L_{task_type}"] = loss.item()
            pbar.set_postfix({k: f"{v:.3f}" for k, v in logs.items() if v != 0.0})

            if args.eval_steps > 0 and (step + 1) % args.eval_steps == 0:
                print(f"\n--- Epoch {epoch} Step {step+1} Evaluation ---")
                best_global_score = _run_evaluation(
                    model=model,
                    dev_loaders=dev_loaders,
                    device=device,
                    epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_global_score=best_global_score,
                    model_path=model_path,
                    args=args
                )

        print(f"\n--- Epoch {epoch} End Evaluation ---")
        best_global_score = _run_evaluation(
            model=model,
            dev_loaders=dev_loaders,
            device=device,
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            best_global_score=best_global_score,
            model_path=model_path,
            args=args
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--hpc_jsonl", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "hpc_dataset.jsonl"))
    parser.add_argument("--mfp_jsonl", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "mfp_dataset.jsonl"))
    parser.add_argument("--tam_jsonl", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "tam_dataset.jsonl"))
    parser.add_argument("--dataset_A_neighbors", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_A_neighbors.json"))
    parser.add_argument("--dataset_B_neighbors", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_B_neighbors.json"))
    parser.add_argument("--neigh_id_key_A", type=str, default="a_id")
    parser.add_argument("--neigh_id_key_B", type=str, default="b_id")

    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--pretrained_name", type=str, default=default_pretrained_name())

    parser.add_argument("--w_hpc", type=float, default=1.0)
    parser.add_argument("--w_mfp", type=float, default=0.2)
    parser.add_argument("--w_tam", type=float, default=0.3)
    parser.add_argument("--tam_pos_weight", type=float, default=0.0, help="Positive-class loss weight for TAM; 0 computes it automatically from the train-set class ratio.")
    parser.add_argument("--mfp_threshold", type=float, default=0.3, help="Sigmoid probability threshold used for MFP evaluation.")
    parser.add_argument("--mfp_max_pos_weight", type=float, default=5.0, help="Maximum pos_weight for each MFP field.")
    parser.add_argument("--use_sani", type=int, choices=[0, 1], default=1, help="Whether to use the SANI neighbor injection module: 1 to enable, 0 to disable.")

    parser.add_argument("--num_fields", type=int, default=10)
    parser.add_argument("--n_emb", type=int, default=256)
    parser.add_argument("--a_emb", type=int, default=256)
    parser.add_argument("--attn_type", type=str, default="softmax")
    parser.add_argument(
        "--aggregation_method",
        type=str,
        choices=AGGREGATION_METHODS,
        default="attention",
        help="Neighbor aggregation method: attention/top1/mean/max."
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_neighbors", type=int, default=3)
    parser.add_argument("--max_neighbor_length", type=int, default=256)
    parser.add_argument("--max_field_length", type=int, default=32)
    parser.add_argument("--num_hard_neg_per_pos", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=1500, help="Evaluate every N steps; 0 evaluates only at the end of each epoch.")

    main(parser.parse_args())
