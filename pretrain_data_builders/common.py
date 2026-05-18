import re
import json
import random
import argparse
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm

SEED = 42
random.seed(SEED)


class BasePretrainingDatasetBuilder:
    def __init__(self, args):
        self.args = args
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    COLUMN_ALIASES = {
        "ID": "Id",
        "id": "Id",
        "name": "Name",
        "affiliation": "Affiliation",
        "Research interests": "Research Interests",
        "research_interests": "Research Interests",
        "papers": "Papers",
        "projects": "Projects"
    }

    FIELD_OUTPUT_NAMES = {
        "Id": "Id",
        "Name": "Name",
        "Affiliation": "Affiliation",
        "Research Interests": "Research Interests",
        "Papers": "Papers",
        "Papers 1": "Papers 1",
        "Papers 2": "Papers 2",
        "Projects": "Projects",
        "Projects 1": "Projects 1",
        "Projects 2": "Projects 2",
        "Candidate Affiliation": "Candidate Affiliation"
    }

    @classmethod
    def normalize_schema_columns(cls, df):
        rename_map = {}
        for col in df.columns:
            canonical = cls.COLUMN_ALIASES.get(col)
            if canonical is not None and canonical not in df.columns:
                rename_map[col] = canonical
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    @classmethod
    def canonical_col(cls, col):
        return cls.COLUMN_ALIASES.get(col, col)

    @classmethod
    def output_field_name(cls, col):
        return cls.FIELD_OUTPUT_NAMES.get(col, col)

    @classmethod
    def field_aliases(cls, field_name):
        canonical = cls.canonical_col(field_name)
        return [cls.output_field_name(canonical)]

    @classmethod
    def all_attr_field_labels(cls):
        labels = list(cls.FIELD_OUTPUT_NAMES.values())
        labels = list(dict.fromkeys(labels))
        labels.sort(key=len, reverse=True)
        return labels

    @staticmethod
    def normalize_text(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    @staticmethod
    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return int(str(x).strip())

    @classmethod
    def build_id_to_row(cls, df, id_col="Id"):
        id_col = cls.canonical_col(id_col)
        out = {}
        for _, row in df.iterrows():
            idx = str(row[id_col])
            out[idx] = row
        return out

    @classmethod
    def build_name_to_ids(cls, df, id_col="Id", name_col="Name"):
        id_col = cls.canonical_col(id_col)
        name_col = cls.canonical_col(name_col)
        name_to_ids = {}
        for _, row in df.iterrows():
            idx = str(row[id_col])
            name = cls.normalize_text(row[name_col])
            name_to_ids.setdefault(name, []).append(idx)
        return name_to_ids

    @classmethod
    def row_to_attr_text_A(cls, row, override_name=None):
        pieces = []
        ordered_fields = ["Id", "Name", "Affiliation", "Research Interests", "Papers", "Projects"]
        for col in ordered_fields:
            if col not in row:
                continue
            val = cls.normalize_text(row[col])
            if col == "Name" and override_name is not None:
                val = override_name
            pieces.append(f"COL {cls.output_field_name(col)} VAL {val}")
        return " ".join(pieces)

    @classmethod
    def row_to_attr_text_B(cls, row, override_name=None):
        pieces = []
        possible_orders = [
            "Id", "Name", "Affiliation",
            "Research Interests",
            "Papers", "Papers 1", "Papers 2",
            "Projects", "Projects 1", "Projects 2"
        ]
        used = set()
        for col in possible_orders:
            if col in row and col not in used:
                val = cls.normalize_text(row[col])
                if col == "Name" and override_name is not None:
                    val = override_name
                pieces.append(f"COL {cls.output_field_name(col)} VAL {val}")
                used.add(col)
        return " ".join(pieces)

    @classmethod
    def load_neighbors_json(cls, path, id_key):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        neigh_map = {}
        for item in data:
            idx = str(item[id_key])
            neigh_map[idx] = {
                "attr": item.get("attr", ""),
                "neighs": item.get("neighs", []),
                "neighs_attr": item.get("neighs_attr", [])
            }
        return neigh_map

    @classmethod
    def extract_field_from_attr(cls, attr_text, field_name):
        attr_text = cls.normalize_text(attr_text)
        if not attr_text:
            return ""
        boundary_labels = "|".join(re.escape(x) for x in cls.all_attr_field_labels())
        for label in cls.field_aliases(field_name):
            pattern = rf"COL\s+{re.escape(label)}\s+VAL\s+(.*?)(?=\s+COL\s+(?:{boundary_labels})\s+VAL|$)"
            m = re.search(pattern, attr_text)
            if m:
                return m.group(1).strip()
        return ""

    @classmethod
    def token_set(cls, text):
        text = cls.normalize_text(text).lower()
        if not text:
            return set()
        toks = re.split(r"[\s,，;；|｜/\\()\[\]{}:_\-\.]+", text)
        toks = [t for t in toks if t]
        return set(toks)

    @staticmethod
    def jaccard_set(s1, s2):
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union > 0 else 0.0

    @classmethod
    def neighbor_overlap_score(cls, neighs_attr_a, neighs_attr_b):
        orgs_a, orgs_b = set(), set()
        kw_a, kw_b = set(), set()

        for x in neighs_attr_a:
            orgs_a.add(cls.extract_field_from_attr(x, "Affiliation"))
            kw_a |= cls.token_set(cls.extract_field_from_attr(x, "Research Interests"))
            kw_a |= cls.token_set(cls.extract_field_from_attr(x, "Papers"))
            kw_a |= cls.token_set(cls.extract_field_from_attr(x, "Papers 1"))
            kw_a |= cls.token_set(cls.extract_field_from_attr(x, "Papers 2"))
            kw_a |= cls.token_set(cls.extract_field_from_attr(x, "Projects"))
            kw_a |= cls.token_set(cls.extract_field_from_attr(x, "Projects 1"))
            kw_a |= cls.token_set(cls.extract_field_from_attr(x, "Projects 2"))

        for x in neighs_attr_b:
            orgs_b.add(cls.extract_field_from_attr(x, "Affiliation"))
            kw_b |= cls.token_set(cls.extract_field_from_attr(x, "Research Interests"))
            kw_b |= cls.token_set(cls.extract_field_from_attr(x, "Papers"))
            kw_b |= cls.token_set(cls.extract_field_from_attr(x, "Papers 1"))
            kw_b |= cls.token_set(cls.extract_field_from_attr(x, "Papers 2"))
            kw_b |= cls.token_set(cls.extract_field_from_attr(x, "Projects"))
            kw_b |= cls.token_set(cls.extract_field_from_attr(x, "Projects 1"))
            kw_b |= cls.token_set(cls.extract_field_from_attr(x, "Projects 2"))

        org_score = cls.jaccard_set(set([o for o in orgs_a if o]), set([o for o in orgs_b if o]))
        kw_score = cls.jaccard_set(kw_a, kw_b)

        return {
            "neighbor_org_overlap": org_score,
            "neighbor_kw_overlap": kw_score,
            "neighbor_overlap_score": 0.5 * org_score + 0.5 * kw_score
        }

    @classmethod
    def attr_overlap_score(cls, attr_a, attr_b):
        org_a = cls.extract_field_from_attr(attr_a, "Affiliation")
        org_b = cls.extract_field_from_attr(attr_b, "Affiliation")
        org_score = 1.0 if org_a and org_b and (org_a in org_b or org_b in org_a) else 0.0

        kw_a = set()
        kw_b = set()
        for field in ["Research Interests", "Papers", "Papers 1", "Papers 2", "Projects", "Projects 1", "Projects 2"]:
            kw_a |= cls.token_set(cls.extract_field_from_attr(attr_a, field))
            kw_b |= cls.token_set(cls.extract_field_from_attr(attr_b, field))

        kw_score = cls.jaccard_set(kw_a, kw_b)
        return {
            "attr_org_overlap": org_score,
            "attr_kw_overlap": kw_score,
            "attr_overlap_score": 0.4 * org_score + 0.6 * kw_score
        }

    @staticmethod
    def save_jsonl(samples, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")


