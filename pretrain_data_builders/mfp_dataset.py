import re
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import BasePretrainingDatasetBuilder

class MFPDatasetBuilder(BasePretrainingDatasetBuilder):
    @classmethod
    def split_research_fields(cls, field_text: str):
        if not field_text:
            return []
        field_text = cls.normalize_text(field_text)
        parts = re.split(r"[|｜;/；、,，]+", field_text)
        parts = [p.strip() for p in parts if p.strip()]
        return parts

    @classmethod
    def extract_field_from_attr_text(cls, attr_text: str, field_name: str):
        return cls.extract_field_from_attr(attr_text, field_name)

    @classmethod
    def build_attr_text_from_row(cls, row, research_fields_override=None):
        ordered_fields = ["Name", "Affiliation", "Research Interests", "Papers", "Projects"]
        pieces = []
        for col in ordered_fields:
            if col not in row:
                continue
            val = cls.normalize_text(row[col])
            if col == "Research Interests" and research_fields_override is not None:
                val = " | ".join(research_fields_override)
            pieces.append(f"COL {cls.output_field_name(col)} VAL {val}")
        return "\n".join(pieces)

    @classmethod
    def build_label_vocab(cls, df: pd.DataFrame, research_col="Research Interests", min_freq=1):
        counter = {}
        for _, row in df.iterrows():
            fields = cls.split_research_fields(cls.normalize_text(row[research_col]))
            for f in fields:
                counter[f] = counter.get(f, 0) + 1
        vocab = [k for k, v in counter.items() if v >= min_freq]
        vocab = sorted(vocab)
        field2id = {f: i for i, f in enumerate(vocab)}
        id2field = {i: f for f, i in field2id.items()}
        return field2id, id2field, counter

    @staticmethod
    def make_local_multi_hot(local_fields, scholar_fields, max_len=10):
        local_fields = list(dict.fromkeys(local_fields))
        if len(local_fields) > max_len:
            local_fields = local_fields[:max_len]
        vec = [1 if f in scholar_fields else 0 for f in local_fields]
        return vec, local_fields

    @classmethod
    def build_global_field_pool(cls, df, research_col):
        fields = []
        for _, row in df.iterrows():
            fields += cls.split_research_fields(cls.normalize_text(row[research_col]))
        return list(dict.fromkeys(fields))

    def corrupt_research_fields(self, scholar_fields, global_field_pool):
        corrupted = scholar_fields[:]
        num_to_corrupt = max(1, int(round(len(scholar_fields) * self.args.mfp_mask_ratio)))
        num_to_corrupt = min(num_to_corrupt, len(scholar_fields))
        selected_indices = sorted(random.sample(range(len(scholar_fields)), num_to_corrupt))

        random_pool = [f for f in global_field_pool if f not in scholar_fields]
        if not random_pool:
            random_pool = global_field_pool[:]

        for idx in selected_indices:
            if random.random() < self.args.mfp_mask_prob:
                corrupted[idx] = self.args.mask_token
            else:
                corrupted[idx] = random.choice(random_pool) if random_pool else self.args.mask_token
        return corrupted, selected_indices

    def run(self):
        input_csv = Path(self.args.input_csv)
        input_neigh_json = Path(self.args.input_neighbors_json)
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_csv).fillna("")
        df = self.normalize_schema_columns(df)
        if self.args.max_samples is not None:
            df = df.head(self.args.max_samples)
        self.args.id_col = self.canonical_col(self.args.id_col)
        self.args.research_col = self.canonical_col(self.args.research_col)

        neigh_map = self.load_neighbors_json(input_neigh_json, self.args.neigh_id_key_A)
        global_field_pool = self.build_global_field_pool(df, self.args.research_col)
        samples = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building MFP dataset"):
            scholar_id = str(row[self.args.id_col])
            scholar_fields = self.split_research_fields(self.normalize_text(row[self.args.research_col]))
            if len(scholar_fields) == 0:
                continue

            if scholar_id not in neigh_map:
                neighbors_attr = []
                neighbors_fields = []
            else:
                neighbors_attr = neigh_map[scholar_id]["neighs_attr"][:self.args.max_neighbors]
                neighbors_fields = []
                for attr in neighbors_attr:
                    if isinstance(attr, dict):
                        field_text = attr.get("Research Interests", "")
                    else:
                        field_text = self.extract_field_from_attr_text(attr, "Research Interests")
                    neighbors_fields += self.split_research_fields(field_text)

            local_fields = scholar_fields + neighbors_fields
            target_multi_hot, local_fields = self.make_local_multi_hot(
                local_fields, scholar_fields, max_len=self.args.max_local_fields
            )
            corrupted_fields, selected_indices = self.corrupt_research_fields(
                scholar_fields,
                global_field_pool
            )
            input_text = self.build_attr_text_from_row(
                row,
                research_fields_override=corrupted_fields
            )

            sample = {
                "id": scholar_id,
                "source": "A",
                "input_text": input_text,
                "candidate_fields": local_fields,
                "target_multi_hot": target_multi_hot,
                "masked_field_indices": selected_indices,
                "corrupted_research_fields": corrupted_fields,
                "original_research_fields": scholar_fields
            }
            samples.append(sample)

        out_jsonl = output_dir / "mfp_dataset.jsonl"
        self.save_jsonl(samples, out_jsonl)
        print(f"Saved {len(samples)} MFP samples to: {out_jsonl}")


