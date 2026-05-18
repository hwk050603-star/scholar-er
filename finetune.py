import os
import json
import random
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer, get_linear_schedule_with_warmup

from model import AGGREGATION_METHODS, SampLayer


SPECIAL_TOKENS = ["COL", "VAL"]
PROJECT_ROOT = Path(__file__).resolve().parent
ATTR_FIELD_NAMES = ["Name", "Affiliation", "Research Interests", "Papers", "Projects"]


def default_pretrained_name():
    local_model_dir = PROJECT_ROOT / "xlm-roberta-base"
    if local_model_dir.exists():
        return str(local_model_dir)
    return "xlm-roberta-base"


def normalize_key(value):
    return " ".join(str(value or "").strip().split()).casefold()


def extract_attr_field(text, field_name):
    text = str(text or "").lstrip("\ufeff")
    marker = f"COL {field_name} VAL "
    start = text.find(marker)
    if start < 0:
        return ""

    start += len(marker)
    next_positions = []
    for name in ATTR_FIELD_NAMES:
        next_marker = f" COL {name} VAL "
        pos = text.find(next_marker, start)
        if pos >= 0:
            next_positions.append(pos)

    end = min(next_positions) if next_positions else len(text)
    return text[start:end].strip()


def load_id_lookup(csv_path):
    lookup = {}
    duplicate_keys = set()

    if not csv_path or not os.path.exists(csv_path):
        print(f"Warning: dataset csv not found for id lookup -> {csv_path}")
        return lookup

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = normalize_key(row.get("Name"))
            affiliation = normalize_key(row.get("Affiliation"))
            entity_id = str(row.get("Id", "")).strip()
            if not name or not affiliation or not entity_id:
                continue

            key = (name, affiliation)
            if key in lookup and lookup[key] != entity_id:
                duplicate_keys.add(key)
                continue
            lookup[key] = entity_id

    print(f"Loaded id lookup from {csv_path}: {len(lookup)} name+affiliation keys")
    if duplicate_keys:
        print(
            "Warning: duplicate name+affiliation keys found in "
            f"{csv_path}: {len(duplicate_keys)}. Keeping the first id for each key."
        )
    return lookup


def find_entity_id(text, lookup):
    name = normalize_key(extract_attr_field(text, "Name"))
    affiliation = normalize_key(extract_attr_field(text, "Affiliation"))
    if not name or not affiliation:
        return None
    return lookup.get((name, affiliation))


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
    print(f"Loaded neighbors from {path}: {len(neigh_map)} entities")
    return neigh_map


def print_neighbor_coverage(name, dataset, neighbors_A, neighbors_B):
    total = len(dataset)
    if total == 0:
        print(f"{name} neighbor coverage A/B: 0.00% / 0.00%")
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

    print(
        f"{name} neighbor coverage A/B: "
        f"{covered_A / total * 100:.2f}% / {covered_B / total * 100:.2f}%"
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]
    else:
        return batch


class ScholarBinaryDataset(Dataset):

    def __init__(self, file_path, left_lookup=None, right_lookup=None, max_samples=None):
        self.samples = []
        self.missing_left_ids = 0
        self.missing_right_ids = 0
        self.left_lookup = left_lookup or {}
        self.right_lookup = right_lookup or {}

        if not os.path.exists(file_path):
            print(f"Warning: file not found -> {file_path}")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                sample = self.parse_line(line, line_id)
                if sample["id_left"] is None and self.left_lookup:
                    sample["id_left"] = find_entity_id(sample["left_text"], self.left_lookup)
                if sample["id_right"] is None and self.right_lookup:
                    sample["id_right"] = find_entity_id(sample["right_text"], self.right_lookup)

                if self.left_lookup and sample["id_left"] is None:
                    self.missing_left_ids += 1
                if self.right_lookup and sample["id_right"] is None:
                    self.missing_right_ids += 1

                self.samples.append({
                    "left_text": sample["left_text"],
                    "right_text": sample["right_text"],
                    "id_left": sample["id_left"],
                    "id_right": sample["id_right"],
                    "label": sample["label"]
                })

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def parse_line(self, line, line_id):
        line = line.lstrip("\ufeff")
        parts = line.split("\t")

        if len(parts) == 5:
            id_left, id_right, left_text, right_text, label_str = [p.strip() for p in parts]
            id_left = id_left or None
            id_right = id_right or None
        elif len(parts) == 3:
            left_text, right_text, label_str = [p.strip() for p in parts]
            id_left, id_right = None, None
        else:
            raise ValueError(
                f"[Line {line_id}] Unable to parse sample. Expected either 3 columns "
                f"'<left_text>\\t<right_text>\\t<label>' or 5 columns "
                f"'<id_left>\\t<id_right>\\t<left_text>\\t<right_text>\\t<label>'.\n"
                f"Raw content: {line[:200]}"
            )

        if label_str not in {"0", "1"}:
            raise ValueError(
                f"[Line {line_id}] Last field is not 0/1.\n"
                f"Detected last field: {label_str}\nRaw content: {line[:200]}"
            )

        return {
            "left_text": left_text,
            "right_text": right_text,
            "id_left": id_left,
            "id_right": id_right,
            "label": int(label_str)
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BinaryCollator:
    def __init__(
        self,
        tokenizer,
        max_length=256,
        max_neighbors=5,
        max_neighbor_length=256,
        neighbors_A=None,
        neighbors_B=None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_neighbors = max_neighbors
        self.max_neighbor_length = max_neighbor_length
        self.neighbors_A = neighbors_A or {}
        self.neighbors_B = neighbors_B or {}

    def _build_pair_input(self, text1, text2):
        t1_ids = self.tokenizer.encode(text1, add_special_tokens=False)
        t2_ids = self.tokenizer.encode(text2, add_special_tokens=False)
        max_t = self.max_length - 4

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

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        return input_ids, attention_mask, (e1_pos, e2_pos)

    def _encode_neighbors(self, neighbors):
        ids, masks = [], []
        for text in (neighbors or [])[:self.max_neighbors]:
            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_neighbor_length,
                padding="max_length"
            )
            ids.append(enc["input_ids"])
            masks.append(enc["attention_mask"])

        if not ids:
            return (
                torch.empty((0, self.max_neighbor_length), dtype=torch.long),
                torch.empty((0, self.max_neighbor_length), dtype=torch.long)
            )

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(masks, dtype=torch.long)
        )

    def __call__(self, batch):
        input_ids, attention_masks, labels, entity_pos_list, x_n = [], [], [], [], []

        for sample in batch:
            ids, mask, pair_pos = self._build_pair_input(sample["left_text"], sample["right_text"])
            id_left = None if sample.get("id_left") is None else str(sample["id_left"])
            id_right = None if sample.get("id_right") is None else str(sample["id_right"])

            neigh1_ids, neigh1_mask = self._encode_neighbors(self.neighbors_A.get(id_left, []))
            neigh2_ids, neigh2_mask = self._encode_neighbors(self.neighbors_B.get(id_right, []))

            input_ids.append(ids)
            attention_masks.append(mask)
            labels.append(sample["label"])
            entity_pos_list.append(pair_pos)
            x_n.append({
                "neigh1_input_ids": neigh1_ids,
                "neigh1_attention_mask": neigh1_mask,
                "neigh2_input_ids": neigh2_ids,
                "neigh2_attention_mask": neigh2_mask
            })

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "entity_pos_list": entity_pos_list,
            "x_n": x_n
        }

class ScholarBinaryClassifier(nn.Module):
    """
    Pair cross-encoder binary classifier with optional SANI neighbor injection.
    """

    def __init__(
        self,
        device,
        pretrained_name="xlm-roberta-base",
        dropout=0.1,
        num_labels=2,
        n_emb=256,
        a_emb=256,
        attn_type="softmax",
        use_sani=True,
        aggregation_method="attention"
    ):
        super().__init__()
        self.device = device
        self.use_sani = bool(use_sani)

        self.language_model = XLMRobertaModel.from_pretrained(pretrained_name)
        self.neighbert = self.language_model
        hidden_size = self.language_model.config.hidden_size

        self.samp_layer = SampLayer(
            a_emb=a_emb,
            n_emb=n_emb,
            hidden_size=hidden_size,
            device=device,
            attn_type=attn_type,
            aggregation_method=aggregation_method
        )
        self.post_neighbor_self_attn = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=self.language_model.config.num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

    def apply_post_neighbor_self_attention(self, hidden_states, attention_mask):
        key_padding_mask = attention_mask == 0
        return self.post_neighbor_self_attn(
            hidden_states,
            src_key_padding_mask=key_padding_mask
        )

    def forward(self, input_ids, attention_mask, labels=None, x_n=None, entity_pos_list=None):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state
        if self.use_sani:
            if x_n is None or entity_pos_list is None:
                raise ValueError("SANI fine-tuning requires x_n and entity_pos_list in each batch.")

            hidden_states = self.samp_layer.inject_pair_neighbors(
                x_n=x_n,
                b_s=input_ids.shape[0],
                xs=hidden_states,
                neighbert=self.neighbert,
                entity_pos_list=entity_pos_list
            )
            hidden_states = self.apply_post_neighbor_self_attention(hidden_states, attention_mask)

        cls_repr = hidden_states[:, 0, :]
        logits = self.classifier(self.dropout(cls_repr))

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return loss, logits

        return logits


def load_model_from_pretrain(model, ckpt_path, device):
    """
    Load language_model/SANI weights from the SER pre-training checkpoint.
    """
    if not ckpt_path:
        print("No pretrain checkpoint provided. Train from scratch.")
        return model

    if not os.path.exists(ckpt_path):
        print(f"Warning: checkpoint not found -> {ckpt_path}")
        return model

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model_state = model.state_dict()
    loadable_state = {}
    skipped_shape = []

    for k, v in state_dict.items():
        new_key = None
        if k.startswith(("language_model.", "samp_layer.", "post_neighbor_self_attn.")):
            new_key = k
        elif k.startswith("encoder."):
            new_key = "language_model." + k[len("encoder."):]

        if new_key is None or new_key not in model_state:
            continue

        if tuple(model_state[new_key].shape) != tuple(v.shape):
            skipped_shape.append((new_key, tuple(v.shape), tuple(model_state[new_key].shape)))
            continue

        loadable_state[new_key] = v

    if len(loadable_state) == 0:
        print("Warning: no loadable language_model/SANI weights were found in the checkpoint; skipping load.")
        return model

    missing, unexpected = model.load_state_dict(loadable_state, strict=False)
    print(f"Loaded pretrain weights from: {ckpt_path}")
    print(f"Loaded keys: {len(loadable_state)}")
    print(f"Missing keys after partial load: {len(missing)}")
    print(f"Unexpected keys after partial load: {len(unexpected)}")
    if skipped_shape:
        print(f"Skipped keys because of shape mismatch: {len(skipped_shape)}")

    return model


@torch.no_grad()
def evaluate(model, dataloader, device, desc="Evaluating"):
    if len(dataloader) == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }

    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_count = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        batch = move_to_device(batch, device)

        loss, logits = model(**batch)

        preds = torch.argmax(logits, dim=1)

        bs = batch["input_ids"].size(0)
        total_loss += loss.item() * bs
        total_count += bs

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch["labels"].detach().cpu().tolist())

    avg_loss = total_loss / max(total_count, 1)
    acc = accuracy_score(all_labels, all_preds) if total_count > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    precision = precision_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)

    model.train()
    return {
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = XLMRobertaTokenizer.from_pretrained(args.pretrained_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    print(f"Use SANI module during fine-tuning: {bool(args.use_sani)}")
    if args.use_sani:
        print(f"Neighbor aggregation method: {args.aggregation_method}")
    left_lookup = load_id_lookup(args.dataset_A) if args.use_sani else {}
    right_lookup = load_id_lookup(args.dataset_B) if args.use_sani else {}
    neighbors_A = load_neighbors_json(args.dataset_A_neighbors, args.neigh_id_key_A) if args.use_sani else {}
    neighbors_B = load_neighbors_json(args.dataset_B_neighbors, args.neigh_id_key_B) if args.use_sani else {}

    train_dataset = ScholarBinaryDataset(
        args.train_path,
        left_lookup=left_lookup,
        right_lookup=right_lookup,
        max_samples=args.max_train_samples
    )
    dev_dataset = ScholarBinaryDataset(
        args.dev_path,
        left_lookup=left_lookup,
        right_lookup=right_lookup,
        max_samples=args.max_dev_samples
    )
    test_dataset = ScholarBinaryDataset(
        args.test_path,
        left_lookup=left_lookup,
        right_lookup=right_lookup,
        max_samples=args.max_test_samples
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Dev size  : {len(dev_dataset)}")
    print(f"Test size : {len(test_dataset)}")
    if args.use_sani:
        print_neighbor_coverage("train", train_dataset, neighbors_A, neighbors_B)
        print_neighbor_coverage("valid", dev_dataset, neighbors_A, neighbors_B)
        print_neighbor_coverage("test", test_dataset, neighbors_A, neighbors_B)

    if len(train_dataset) == 0:
        print("Training set is empty. Exit.")
        return

    collator = BinaryCollator(
        tokenizer,
        max_length=args.max_length,
        max_neighbors=args.max_neighbors,
        max_neighbor_length=args.max_neighbor_length,
        neighbors_A=neighbors_A,
        neighbors_B=neighbors_B
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator
    )

    model = ScholarBinaryClassifier(
        device=device,
        pretrained_name=args.pretrained_name,
        dropout=args.dropout,
        num_labels=2,
        n_emb=args.n_emb,
        a_emb=args.a_emb,
        attn_type=args.attn_type,
        use_sani=bool(args.use_sani),
        aggregation_method=args.aggregation_method
    )
    model.language_model.resize_token_embeddings(len(tokenizer))

    model = load_model_from_pretrain(model, args.pretrained_ckpt, device)

    if args.freeze_encoder:
        for p in model.language_model.parameters():
            p.requires_grad = False
        print("Encoder frozen.")

    model = model.to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pth"

    best_dev_f1 = -1.0
    best_dev_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        epoch_preds = []
        epoch_labels = []

        print(f"\nEpoch {epoch}/{args.epochs}")
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, device)

            optimizer.zero_grad()

            loss, logits = model(**batch)

            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=1)

            epoch_preds.extend(preds.detach().cpu().tolist())
            epoch_labels.extend(batch["labels"].detach().cpu().tolist())
            running_loss += loss.item() * batch["input_ids"].size(0)

            train_acc = accuracy_score(epoch_labels, epoch_preds) if len(epoch_labels) > 0 else 0.0
            train_f1 = f1_score(epoch_labels, epoch_preds, average="binary", pos_label=1, zero_division=0)

            if args.print_steps > 0 and step % args.print_steps == 0:
                print(
                    f"step: {step}, "
                    f"loss: {loss.item():.12f}"
                )

        train_loss = running_loss / max(len(train_dataset), 1)
        train_acc = accuracy_score(epoch_labels, epoch_preds)
        train_f1 = f1_score(epoch_labels, epoch_preds, average="binary", pos_label=1, zero_division=0)

        print(f"\n[Epoch {epoch}] Train Loss: {train_loss:.4f}")

        dev_metrics = evaluate(model, dev_loader, device, desc="Dev")
        print(
            f"[Epoch {epoch}] Dev Loss: {dev_metrics['loss']:.4f} | "
            f"Acc: {dev_metrics['acc']:.4f} | "
            f"F1: {dev_metrics['f1']:.4f} | "
            f"P: {dev_metrics['precision']:.4f} | "
            f"R: {dev_metrics['recall']:.4f}"
        )

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_dev_acc = dev_metrics["acc"]
            test_metrics_for_best = evaluate(model, test_loader, device, desc="Test@BestDev")
            print(
                f"[Epoch {epoch}] Test@BestDev Loss: {test_metrics_for_best['loss']:.4f} | "
                f"Acc: {test_metrics_for_best['acc']:.4f} | "
                f"F1: {test_metrics_for_best['f1']:.4f} | "
                f"P: {test_metrics_for_best['precision']:.4f} | "
                f"R: {test_metrics_for_best['recall']:.4f}"
            )

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dev_f1": best_dev_f1,
                "best_dev_acc": best_dev_acc,
                "args": vars(args),
                "dev_metrics": dev_metrics,
                "test_metrics": test_metrics_for_best
            }
            torch.save(ckpt, model_path)
            print(f"==> New best model saved to {model_path}.\n")

    if os.path.exists(model_path):
        best_ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device, desc="Test")
    print("\n========== Final Test ==========")
    print(f"Best Dev F1 : {best_dev_f1:.4f}")
    print(f"Best Dev Acc: {best_dev_acc:.4f}")
    print(f"Test Loss   : {test_metrics['loss']:.4f}")
    print(f"Test Acc    : {test_metrics['acc']:.4f}")
    print(f"Test F1     : {test_metrics['f1']:.4f}")
    print(f"Test Prec   : {test_metrics['precision']:.4f}")
    print(f"Test Recall : {test_metrics['recall']:.4f}")
    print("================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default=str(PROJECT_ROOT / "blocking" / "train.txt"))
    parser.add_argument("--dev_path", type=str, default=str(PROJECT_ROOT / "blocking" / "valid.txt"))
    parser.add_argument("--test_path", type=str, default=str(PROJECT_ROOT / "blocking" / "test.txt"))
    parser.add_argument("--dataset_A", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_A_aug.csv"))
    parser.add_argument("--dataset_B", type=str, default=str(PROJECT_ROOT / "blocking" / "dataset_B_aug.csv"))
    parser.add_argument("--dataset_A_neighbors", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_A_neighbors.json"))
    parser.add_argument("--dataset_B_neighbors", type=str, default=str(PROJECT_ROOT / "pretrain" / "dataset" / "dataset_B_neighbors.json"))
    parser.add_argument("--neigh_id_key_A", type=str, default="a_id")
    parser.add_argument("--neigh_id_key_B", type=str, default="b_id")

    parser.add_argument("--pretrained_ckpt", type=str, default=str(PROJECT_ROOT / "checkpoints" / "model.pth"))
    parser.add_argument("--pretrained_name", type=str, default=default_pretrained_name())

    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "finetune_checkpoints"))

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_neighbors", type=int, default=3)
    parser.add_argument("--max_neighbor_length", type=int, default=256)
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
    parser.add_argument("--use_sani", type=int, choices=[0, 1], default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--print_steps", type=int, default=10, help="Print training loss every N steps; 0 disables step-level logging.")

    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_dev_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)

    args = parser.parse_args()
    train(args)
