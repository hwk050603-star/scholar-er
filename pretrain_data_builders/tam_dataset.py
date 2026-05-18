import re
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import BasePretrainingDatasetBuilder

class TAMDatasetBuilder(BasePretrainingDatasetBuilder):
    @classmethod
    def exclude_org_rows(cls, df, org_keyword="计算技术研究所", org_col="Affiliation"):
        if org_col not in df.columns:
            return df
        org_series = df[org_col].fillna("").astype(str)
        keep_mask = ~org_series.str.contains(org_keyword, regex=False)
        return df[keep_mask].copy()

    @classmethod
    def build_candidate_affiliation_attr(cls, name, affiliation_text):
        name = cls.normalize_text(name)
        affiliation_text = cls.normalize_text(affiliation_text)
        return f"COL Name VAL {name} COL Candidate Affiliation VAL {affiliation_text}"

    @classmethod
    def strip_time_from_segment(cls, seg: str):
        seg = cls.normalize_text(seg)
        if not seg:
            return ""
        seg = re.sub(r"[（(][^（）()]{0,60}[)）]\s*$", "", seg).strip()
        return seg

    @classmethod
    def parse_timeline_text(cls, timeline_text: str):
        timeline_text = cls.normalize_text(timeline_text)
        if not timeline_text:
            return []
        parts = [p.strip() for p in timeline_text.split("|") if p.strip()]
        parts = [cls.strip_time_from_segment(p) for p in parts]
        parts = [p for p in parts if p]
        seen = set()
        out = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    @classmethod
    def load_partial_work_history_csv(cls, path):
        rows = []
        name_to_rows = {}
        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        if not lines:
            return rows, name_to_rows

        header = lines[0].strip().lstrip("\ufeff").split(",")
        if len(header) < 3 or header[0] != "Id" or header[1] != "Name" or header[2] != "Work History":
            raise ValueError(f"Unexpected partial_work_history.csv header: {lines[0].strip()}")

        for line in lines[1:]:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split(",", 2)
            if len(parts) < 3:
                parts = parts + [""] * (3 - len(parts))
            item = {
                "Id": cls.normalize_text(parts[0]),
                "Name": cls.normalize_text(parts[1]),
                "Work History": cls.normalize_text(parts[2]),
                "affiliations": cls.parse_timeline_text(parts[2])
            }
            rows.append(item)
            name_to_rows.setdefault(item["Name"], []).append(item)
        return rows, name_to_rows

    @classmethod
    def choose_best_timeline_record(cls, name, current_org, name_to_rows):
        name = cls.normalize_text(name)
        current_org = cls.normalize_text(current_org)
        candidates = name_to_rows.get(name, [])
        if not candidates:
            return None
        for item in candidates:
            affs = item.get("affiliations", [])
            if current_org in affs:
                return item
        for item in candidates:
            for aff in item.get("affiliations", []):
                if current_org and aff and (current_org in aff or aff in current_org):
                    return item
        return candidates[0]

    @classmethod
    def extract_university_name(cls, text: str):
        text = cls.normalize_text(text)
        if not text:
            return ""
        m = re.search(r"(.+?大学)", text)
        if m:
            return m.group(1).strip()
        return ""

    @classmethod
    def extract_college_like_fragment(cls, text: str):
        text = cls.normalize_text(text)
        if not text:
            return ""
        patterns = [
            r"([^，,。；;]*学院)",
            r"([^，,。；;]*研究院)",
            r"([^，,。；;]*系)",
            r"([^，,。；;]*中心)",
            r"([^，,。；;]*学部)",
            r"([^，,。；;]*实验室)"
        ]
        for p in patterns:
            m = re.search(p, text)
            if m:
                return m.group(1).strip()
        return ""

    @classmethod
    def is_same_university_diff_college(cls, org1: str, org2: str):
        org1 = cls.normalize_text(org1)
        org2 = cls.normalize_text(org2)
        if not org1 or not org2 or org1 == org2:
            return False
        u1 = cls.extract_university_name(org1)
        u2 = cls.extract_university_name(org2)
        if not u1 or not u2 or u1 != u2:
            return False
        c1 = cls.extract_college_like_fragment(org1)
        c2 = cls.extract_college_like_fragment(org2)
        if c1 and c2 and c1 != c2:
            return True
        return org1 != org2

    @classmethod
    def build_global_affiliation_pool(cls, timeline_rows):
        pool = []
        for item in timeline_rows:
            name = item["Name"]
            affs = item.get("affiliations", [])
            for aff in affs:
                if aff:
                    pool.append({"name": name, "affiliation": aff})
        return pool

    @classmethod
    def build_affiliation_sampler(cls, global_aff_pool):
        unique_items = []
        seen = set()
        for item in global_aff_pool:
            name = cls.normalize_text(item.get("name", ""))
            aff = cls.normalize_text(item.get("affiliation", ""))
            if not aff:
                continue
            key = (name, aff)
            if key in seen:
                continue
            seen.add(key)
            university = cls.extract_university_name(aff)
            college = cls.extract_college_like_fragment(aff)
            unique_items.append(
                {
                    "name": name,
                    "affiliation": aff,
                    "university": university,
                    "college": college,
                }
            )

        by_university = {}
        diff_university_items = []
        no_university_items = []
        for item in unique_items:
            university = item["university"]
            if university:
                by_university.setdefault(university, []).append(item)
                diff_university_items.append(item)
            else:
                no_university_items.append(item)

        random.shuffle(unique_items)
        random.shuffle(diff_university_items)
        random.shuffle(no_university_items)
        for items in by_university.values():
            random.shuffle(items)

        return {
            "all": unique_items,
            "by_university": by_university,
            "with_university": diff_university_items,
            "without_university": no_university_items,
        }

    @classmethod
    def is_valid_negative_item(cls, item, current_name, current_org, own_history_affs):
        if not item:
            return False
        cand_name = item["name"]
        cand_aff = item["affiliation"]
        if not cand_aff:
            return False
        if cand_name == current_name:
            return False
        if cand_aff == current_org:
            return False
        if cand_aff in own_history_affs:
            return False
        return True

    @classmethod
    def sample_from_items(cls, items, current_name, current_org, own_history_affs, max_attempts=96):
        if not items:
            return None
        attempts = min(max_attempts, len(items))
        for _ in range(attempts):
            item = random.choice(items)
            if cls.is_valid_negative_item(item, current_name, current_org, own_history_affs):
                return item["affiliation"]
        for item in items[:max_attempts]:
            if cls.is_valid_negative_item(item, current_name, current_org, own_history_affs):
                return item["affiliation"]
        return None

    @classmethod
    def load_same_name_tam_negative_pools(cls, path):
        a_to_b = {}
        b_to_a = {}
        if path is None or str(path).strip() == "":
            return a_to_b, b_to_a

        path = Path(path)
        if not path.exists():
            print(f"Warning: same-name negatives file not found: {path}")
            return a_to_b, b_to_a

        df = pd.read_csv(path).fillna("")
        required = {"id_left", "id_right"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}, got {df.columns.tolist()}"
            )

        if "label" in df.columns:
            df = df[df["label"].astype(str) == "0"]

        for _, row in df.iterrows():
            left = cls.normalize_text(row["id_left"])
            right = cls.normalize_text(row["id_right"])
            if not left or not right:
                continue
            a_to_b.setdefault(left, []).append(right)
            b_to_a.setdefault(right, []).append(left)

        for pool in (a_to_b, b_to_a):
            for key, ids in pool.items():
                pool[key] = list(dict.fromkeys(ids))

        print(
            "Loaded TAM same-name negative pools: "
            f"A->B anchors={len(a_to_b)}, pairs={sum(len(v) for v in a_to_b.values())}; "
            f"B->A anchors={len(b_to_a)}, pairs={sum(len(v) for v in b_to_a.values())}"
        )
        return a_to_b, b_to_a

    def build_positive_samples_for_df(self, df, source_name, name_to_timeline_rows, neigh_map):
        id2row = self.build_id_to_row(df)
        samples = []

        for entity_id, row in tqdm(id2row.items(), total=len(id2row), desc=f"Building TAM positive samples for {source_name}"):
            name = self.normalize_text(row.get("Name", ""))
            current_org = self.normalize_text(row.get("Affiliation", ""))
            if not name or not current_org:
                continue

            timeline_item = self.choose_best_timeline_record(name, current_org, name_to_timeline_rows)
            if timeline_item is None:
                continue

            history_affs = timeline_item.get("affiliations", [])
            pos_candidates = [x for x in history_affs if x and x != current_org]
            if len(pos_candidates) == 0:
                continue

            random.shuffle(pos_candidates)
            chosen = pos_candidates[:self.args.num_pos_per_entity]
            entity_attr = self.row_to_attr_text_A(row) if source_name == "A" else self.row_to_attr_text_B(row)
            entity_attr = re.sub(r"^COL\s+Id\s+VAL\s+\S+\s*", "", entity_attr).strip()

            for idx, cand_aff in enumerate(chosen):
                sample = {
                    "id": entity_id,
                    "source": source_name,
                    "entity_name": name,
                    "entity_attr": entity_attr,
                    "candidate_affiliation_text": cand_aff,
                    "candidate_affiliation_attr": self.build_candidate_affiliation_attr(name, cand_aff),
                    "label": 1
                }
                samples.append(sample)

        return samples

    def sample_easy_negative(self, current_name, current_org, own_history_affs, affiliation_sampler):
        current_name = self.normalize_text(current_name)
        current_org = self.normalize_text(current_org)
        own_history_affs = set(self.normalize_text(x) for x in own_history_affs if self.normalize_text(x))
        current_uni = self.extract_university_name(current_org)

        if current_uni:
            items = affiliation_sampler["with_university"]
            attempts = min(128, len(items))
            for _ in range(attempts):
                item = random.choice(items)
                if item["university"] == current_uni:
                    continue
                if self.is_valid_negative_item(item, current_name, current_org, own_history_affs):
                    return item["affiliation"]

        return self.sample_from_items(
            affiliation_sampler["all"],
            current_name,
            current_org,
            own_history_affs,
        )

    def sample_hard_negative(self, current_name, current_org, own_history_affs, affiliation_sampler):
        current_name = self.normalize_text(current_name)
        current_org = self.normalize_text(current_org)
        own_history_affs = set(self.normalize_text(x) for x in own_history_affs if self.normalize_text(x))
        current_uni = self.extract_university_name(current_org)
        if not current_uni:
            return None

        candidates = affiliation_sampler["by_university"].get(current_uni, [])
        current_college = self.extract_college_like_fragment(current_org)
        if current_college:
            candidates = [
                item for item in candidates
                if item["college"] and item["college"] != current_college
            ] or candidates

        cand_aff = self.sample_from_items(
            candidates,
            current_name,
            current_org,
            own_history_affs,
        )
        if cand_aff is None:
            return None
        if not self.is_same_university_diff_college(current_org, cand_aff):
            return None
        return cand_aff

    def sample_same_name_negative(
        self,
        entity_id,
        current_org,
        own_history_affs,
        same_name_pool,
        counterpart_id2row,
        name_to_timeline_rows
    ):
        current_org = self.normalize_text(current_org)
        own_history_affs = set(self.normalize_text(x) for x in own_history_affs if self.normalize_text(x))

        candidate_ids = list(same_name_pool.get(str(entity_id), []))
        random.shuffle(candidate_ids)

        candidate_affs = []
        for other_id in candidate_ids:
            other_row = counterpart_id2row.get(str(other_id))
            if other_row is None:
                continue

            other_name = self.normalize_text(other_row.get("Name", ""))
            other_current_org = self.normalize_text(other_row.get("Affiliation", ""))
            if not other_name:
                continue

            timeline_item = self.choose_best_timeline_record(
                other_name,
                other_current_org,
                name_to_timeline_rows
            )
            if timeline_item is not None:
                candidate_affs.extend(timeline_item.get("affiliations", []))
            if other_current_org:
                candidate_affs.append(other_current_org)

        candidate_affs = list(dict.fromkeys(self.normalize_text(x) for x in candidate_affs if self.normalize_text(x)))
        random.shuffle(candidate_affs)
        for cand_aff in candidate_affs:
            if cand_aff == current_org:
                continue
            if cand_aff in own_history_affs:
                continue
            return cand_aff
        return None

    def build_negative_samples_for_df(
        self,
        df,
        source_name,
        name_to_timeline_rows,
        affiliation_sampler,
        neigh_map,
        same_name_pool,
        counterpart_id2row
    ):
        id2row = self.build_id_to_row(df)
        samples = []

        for entity_id, row in tqdm(id2row.items(), total=len(id2row), desc=f"Building TAM negative samples for {source_name}"):
            name = self.normalize_text(row.get("Name", ""))
            current_org = self.normalize_text(row.get("Affiliation", ""))
            if not name or not current_org:
                continue

            timeline_item = self.choose_best_timeline_record(name, current_org, name_to_timeline_rows)
            if timeline_item is None:
                continue

            own_history_affs = timeline_item.get("affiliations", [])
            entity_attr = self.row_to_attr_text_A(row) if source_name == "A" else self.row_to_attr_text_B(row)
            entity_attr = re.sub(r"^COL\s+Id\s+VAL\s+\S+\s*", "", entity_attr).strip()
            used_affs = set()

            for i in range(self.args.num_same_name_neg_per_entity):
                cand_aff = self.sample_same_name_negative(
                    entity_id=entity_id,
                    current_org=current_org,
                    own_history_affs=own_history_affs,
                    same_name_pool=same_name_pool,
                    counterpart_id2row=counterpart_id2row,
                    name_to_timeline_rows=name_to_timeline_rows
                )
                if cand_aff is None or cand_aff in used_affs:
                    continue
                used_affs.add(cand_aff)
                sample = {
                    "id": entity_id,
                    "source": source_name,
                    "entity_name": name,
                    "entity_attr": entity_attr,
                    "candidate_affiliation_text": cand_aff,
                    "candidate_affiliation_attr": self.build_candidate_affiliation_attr(name, cand_aff),
                    "label": 0
                }
                samples.append(sample)

            for i in range(self.args.num_hard_neg_per_entity):
                cand_aff = self.sample_hard_negative(name, current_org, own_history_affs, affiliation_sampler)
                if cand_aff is None or cand_aff in used_affs:
                    continue
                used_affs.add(cand_aff)
                sample = {
                    "id": entity_id,
                    "source": source_name,
                    "entity_name": name,
                    "entity_attr": entity_attr,
                    "candidate_affiliation_text": cand_aff,
                    "candidate_affiliation_attr": self.build_candidate_affiliation_attr(name, cand_aff),
                    "label": 0
                }
                samples.append(sample)

            for i in range(self.args.num_easy_neg_per_entity):
                cand_aff = self.sample_easy_negative(name, current_org, own_history_affs, affiliation_sampler)
                if cand_aff is None or cand_aff in used_affs:
                    continue
                used_affs.add(cand_aff)
                sample = {
                    "id": entity_id,
                    "source": source_name,
                    "entity_name": name,
                    "entity_attr": entity_attr,
                    "candidate_affiliation_text": cand_aff,
                    "candidate_affiliation_attr": self.build_candidate_affiliation_attr(name, cand_aff),
                    "label": 0
                }
                samples.append(sample)

        return samples

    def run(self):
        df_A = self.normalize_schema_columns(pd.read_csv(self.args.dataset_A).fillna(""))
        df_B = self.normalize_schema_columns(pd.read_csv(self.args.dataset_B).fillna(""))

        df_A = self.exclude_org_rows(df_A, org_keyword=self.args.exclude_org_keyword, org_col="Affiliation")
        df_B = self.exclude_org_rows(df_B, org_keyword=self.args.exclude_org_keyword, org_col="Affiliation")

        if self.args.max_samples is not None:
            df_A = df_A.head(self.args.max_samples)
            df_B = df_B.head(self.args.max_samples)

        self.args.id_col_A = self.canonical_col(self.args.id_col_A)
        self.args.id_col_B = self.canonical_col(self.args.id_col_B)
        if self.args.id_col_A not in df_A.columns:
            raise ValueError(f"{self.args.id_col_A} not found in dataset_A columns: {df_A.columns.tolist()}")
        if self.args.id_col_B not in df_B.columns:
            raise ValueError(f"{self.args.id_col_B} not found in dataset_B columns: {df_B.columns.tolist()}")

        if self.args.id_col_A != "Id":
            df_A = df_A.rename(columns={self.args.id_col_A: "Id"})
        if self.args.id_col_B != "Id":
            df_B = df_B.rename(columns={self.args.id_col_B: "Id"})

        neigh_A = self.load_neighbors_json(self.args.dataset_A_neighbors, self.args.neigh_id_key_A)
        neigh_B = self.load_neighbors_json(self.args.dataset_B_neighbors, self.args.neigh_id_key_B)
        timeline_rows, name_to_timeline_rows = self.load_partial_work_history_csv(
            self.args.partial_work_history_csv
        )
        global_aff_pool = self.build_global_affiliation_pool(timeline_rows)
        affiliation_sampler = self.build_affiliation_sampler(global_aff_pool)
        same_name_a_to_b, same_name_b_to_a = self.load_same_name_tam_negative_pools(
            self.args.same_name_negatives_csv
        )
        id2row_A = self.build_id_to_row(df_A)
        id2row_B = self.build_id_to_row(df_B)

        pos_A = self.build_positive_samples_for_df(df_A, "A", name_to_timeline_rows, neigh_A)
        pos_B = self.build_positive_samples_for_df(df_B, "B", name_to_timeline_rows, neigh_B)
        neg_A = self.build_negative_samples_for_df(
            df_A,
            "A",
            name_to_timeline_rows,
            affiliation_sampler,
            neigh_A,
            same_name_a_to_b,
            id2row_B
        )
        neg_B = self.build_negative_samples_for_df(
            df_B,
            "B",
            name_to_timeline_rows,
            affiliation_sampler,
            neigh_B,
            same_name_b_to_a,
            id2row_A
        )

        all_samples = pos_A + pos_B + neg_A + neg_B
        random.shuffle(all_samples)
        random.shuffle(all_samples)

        out_path = Path(self.args.output_dir) / "tam_dataset.jsonl"
        self.save_jsonl(all_samples, out_path)
        print(f"Saved TAM dataset to: {out_path}")
        print(f"Total samples: {len(all_samples)}")


