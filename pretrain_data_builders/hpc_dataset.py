import re
import random
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
from tqdm import tqdm

from .common import BasePretrainingDatasetBuilder

class HPCDatasetBuilder(BasePretrainingDatasetBuilder):
    @staticmethod
    def _dedup_keep_order(items):
        seen = set()
        out = []
        for item in items:
            item = str(item).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    @staticmethod
    def _compact_name_text(text):
        return re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", str(text)).lower()

    @classmethod
    def english_name_variants(cls, name: str):
        name = cls.normalize_text(name)
        parts = re.findall(r"[A-Za-z]+", name)
        if len(parts) < 2:
            return [name] if name else []

        first, last = parts[0], parts[-1]
        variants = [
            f"{first} {last}",
            f"{last} {first}",
            f"{first} {last[0]}.",
            f"{first[0]}. {last}",
            f"{last} {first[0]}.",
            f"{last[0]}. {first}",
            f"{first}{last}",
            f"{last}{first}"
        ]
        variants += [v.title() for v in variants]
        return cls._dedup_keep_order([name] + variants)

    @classmethod
    def chinese_name_candidates_from_text(cls, *texts):
        joined = " ".join(cls.normalize_text(t) for t in texts if cls.normalize_text(t))
        if not joined:
            return []
        candidates = re.findall(r"[\u4e00-\u9fff]{2,4}", joined[:1200])
        stopwords = {
            "中国", "国家", "自然", "科学", "基金", "项目", "研究", "大学", "学院",
            "统计", "决策", "论坛", "经济", "模型", "基于", "发展", "周期", "测度",
            "信息", "系统", "工程", "技术", "理论", "方法", "数据"
        }
        return cls._dedup_keep_order([c for c in candidates if c not in stopwords])

    @classmethod
    def matched_chinese_name_variants(cls, english_names, *texts):
        candidates = cls.chinese_name_candidates_from_text(*texts)
        if not candidates:
            return []
        try:
            from pypinyin import lazy_pinyin
        except ImportError:
            return []

        english_norms = set()
        for name in english_names:
            for variant in cls.english_name_variants(name):
                compact = cls._compact_name_text(variant)
                if compact:
                    english_norms.add(compact)

        matched = []
        for cand in candidates:
            pys = lazy_pinyin(cand)
            if not pys:
                continue
            pinyin_forms = {
                "".join(pys).lower(),
                "".join(pys[::-1]).lower()
            }
            if english_norms & pinyin_forms:
                matched.append(cand)
        return cls._dedup_keep_order(matched)

    def hpc_name_variants(self, row_A, row_B):
        name_A = self.normalize_text(row_A.get("Name", ""))
        name_B = self.normalize_text(row_B.get("Name", ""))
        text_fields = []
        for row in [row_A, row_B]:
            for field in ["Papers", "Papers 1", "Papers 2", "Projects", "Projects 1", "Projects 2"]:
                if field in row:
                    text_fields.append(row.get(field, ""))

        chinese_variants = self.matched_chinese_name_variants(
            [name_A, name_B],
            *text_fields
        )
        english_variants = []
        for name in [name_A, name_B]:
            english_variants += self.english_name_variants(name)

        # Prefer grounded Chinese names when available, then abbreviated English forms,
        # then the original source names.
        return self._dedup_keep_order(chinese_variants + english_variants + [name_A, name_B])

    def choose_hpc_override_name(self, row_A, row_B):
        variants = self.hpc_name_variants(row_A, row_B)
        if not variants:
            return self.normalize_text(row_A.get("Name", ""))
        return random.choice(variants)

    @classmethod
    def chinese_name_to_pinyin_variants(cls, name: str):
        name = cls.normalize_text(name)
        if not name:
            return []
        try:
            from pypinyin import lazy_pinyin
            pys = lazy_pinyin(name)
            if len(pys) == 0:
                return []
            full_space = " ".join(pys)
            reverse_space = " ".join(pys[::-1])
            full_join = "".join(pys)
            reverse_join = "".join(pys[::-1])
            variants = [full_space, reverse_space, full_join, reverse_join]
            variants += [v.title() for v in variants]
            seen = set()
            out = []
            for v in variants:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            return out
        except ImportError:
            return []

    def build_scholar_item(
        self,
        row,
        entity_id,
        source,
        neigh_map,
        override_name=None,
        include_source=True,
        include_name=True,
        include_id_in_attr=True,
        include_neighbors=True
    ):
        name = self.normalize_text(row["Name"])
        attr = (
            self.row_to_attr_text_A(row, override_name=override_name)
            if source == "A"
            else self.row_to_attr_text_B(row, override_name=override_name)
        )
        if not include_id_in_attr:
            attr = re.sub(r"^COL\s+Id\s+VAL\s+\S+\s*", "", attr).strip()
        item = {
            "id": entity_id,
            "attr": attr
        }
        if include_neighbors:
            item["neighbors_attr"] = neigh_map.get(entity_id, {}).get("neighs_attr", [])[:self.args.max_neighbors]
        if include_source:
            item["source"] = source
        if include_name:
            item["name"] = override_name if override_name is not None else name
        return item

    def build_hpc_item(self, row, entity_id, source, neigh_map, override_name=None):
        return self.build_scholar_item(
            row=row,
            entity_id=entity_id,
            source=source,
            neigh_map=neigh_map,
            override_name=override_name,
            include_source=True,
            include_name=not self.args.compact_hpc_json,
            include_id_in_attr=not self.args.drop_id_from_hpc_attr,
            include_neighbors=not self.args.omit_hpc_neighbors
        )

    @classmethod
    def load_same_name_negative_pool(cls, path):
        if path is None or str(path).strip() == "":
            return {}
        path = Path(path)
        if not path.exists():
            print(f"Warning: same-name negatives file not found: {path}")
            return {}

        df = pd.read_csv(path).fillna("")
        required = {"id_left", "id_right"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}, got {df.columns.tolist()}"
            )

        pool = {}
        for _, row in df.iterrows():
            id_left = cls.normalize_text(row["id_left"])
            id_right = cls.normalize_text(row["id_right"])
            if not id_left or not id_right:
                continue
            pool.setdefault(id_left, []).append(id_right)

        for id_left, ids in pool.items():
            pool[id_left] = list(dict.fromkeys(ids))
        print(f"Loaded same-name negative pool: anchors={len(pool)}, pairs={sum(len(v) for v in pool.values())}")
        return pool

    @classmethod
    def load_blocking_candidate_pool(cls, path):
        if path is None or str(path).strip() == "":
            return {}
        path = Path(path)
        if not path.exists():
            print(f"Warning: blocking candidates file not found: {path}")
            return {}

        df = pd.read_csv(path).fillna("")
        required = {"id_left", "id_right", "score"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}, got {df.columns.tolist()}"
            )

        if "label" in df.columns:
            df = df[df["label"].astype(str) == "0"]

        rows_by_a = {}
        for _, row in df.iterrows():
            id_left = cls.normalize_text(row["id_left"])
            id_right = cls.normalize_text(row["id_right"])
            if not id_left or not id_right:
                continue
            try:
                score = float(row["score"])
            except Exception:
                score = 0.0
            rows_by_a.setdefault(id_left, []).append((id_right, score))

        pool = {}
        for id_left, rows in rows_by_a.items():
            rows.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            ids = []
            for id_right, _score in rows:
                if id_right in seen:
                    continue
                seen.add(id_right)
                ids.append(id_right)
            pool[id_left] = ids

        print(f"Loaded blocking candidate pool: anchors={len(pool)}, pairs={sum(len(v) for v in pool.values())}")
        return pool

    def score_hard_negative(self, anchor_attr, neg_attr, anchor_neighbors, neg_neighbors):
        attr_score = self.attr_overlap_score(anchor_attr, neg_attr)
        neigh_score = self.neighbor_overlap_score(anchor_neighbors, neg_neighbors)
        return {
            **attr_score,
            **neigh_score,
            "hard_negative_score": (
                0.65 * attr_score["attr_overlap_score"]
                + 0.35 * neigh_score["neighbor_overlap_score"]
            )
        }

    def add_candidates(self, target, candidates, id_A, pos_B, mapped_pairs, id2row_B):
        seen = set(target)
        for bid in candidates:
            bid = str(bid)
            if bid in seen:
                continue
            if bid not in id2row_B:
                continue
            if bid == pos_B:
                continue
            if (id_A, bid) in mapped_pairs:
                continue
            target.append(bid)
            seen.add(bid)

    def sample_easy_negative_ids(self, id_A, pos_B, mapped_pairs, id2row_B, excluded_ids):
        excluded = set(str(x) for x in excluded_ids)
        excluded.add(str(pos_B))
        all_ids = list(id2row_B.keys())
        random.shuffle(all_ids)

        easy_ids = []
        for bid in all_ids:
            bid = str(bid)
            if bid in excluded:
                continue
            if (id_A, bid) in mapped_pairs:
                continue
            easy_ids.append(bid)
            if len(easy_ids) >= self.args.num_easy_neg_per_pos:
                break
        return easy_ids

    def build_pair_classification_samples(self, df_A, df_B, mapping_df, neigh_A, neigh_B):
        id2row_A = self.build_id_to_row(df_A)
        id2row_B = self.build_id_to_row(df_B)
        name_to_ids_B = self.build_name_to_ids(df_B)

        mapped_pairs = set()
        for _, row in mapping_df.iterrows():
            a = str(self.safe_int(row["id_A"]))
            b = str(self.safe_int(row["id_B"]))
            mapped_pairs.add((a, b))

        same_name_negative_pool = self.load_same_name_negative_pool(
            self.args.same_name_negatives_csv
        )
        blocking_candidate_pool = self.load_blocking_candidate_pool(
            self.args.blocking_candidates_csv
        )
        samples = []

        for _, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc="Building HPC pair samples"):
            id_A = str(self.safe_int(row["id_A"]))
            pos_B = str(self.safe_int(row["id_B"]))

            if id_A not in id2row_A or pos_B not in id2row_B:
                continue

            row_A = id2row_A[id_A]
            row_pos_B = id2row_B[pos_B]
            override_name = (
                self.choose_hpc_override_name(row_A, row_pos_B)
                if self.args.use_pinyin_aug
                else None
            )
            name_variants = self.hpc_name_variants(row_A, row_pos_B)
            neighs_A = neigh_A.get(id_A, {}).get("neighs_attr", [])[:self.args.max_neighbors]
            attr_A = self.row_to_attr_text_A(row_A)

            hard_candidates = []
            candidate_sources = {}

            # P0: curated same-name / pinyin-same-name negatives.
            before = len(hard_candidates)
            self.add_candidates(
                hard_candidates,
                same_name_negative_pool.get(id_A, []),
                id_A,
                pos_B,
                mapped_pairs,
                id2row_B
            )
            for bid in hard_candidates[before:]:
                candidate_sources[bid] = 3

            # P1: B-side exact same-name non-matches.
            if len(hard_candidates) < self.args.num_hard_neg_per_pos:
                before = len(hard_candidates)
                for nv in name_variants:
                    self.add_candidates(
                        hard_candidates,
                        name_to_ids_B.get(nv, []),
                        id_A,
                        pos_B,
                        mapped_pairs,
                        id2row_B
                    )
                for bid in hard_candidates[before:]:
                    candidate_sources[bid] = 2

            # P3: high-score blocking candidates, then override their names for HPC.
            if len(hard_candidates) < self.args.num_hard_neg_per_pos:
                before = len(hard_candidates)
                self.add_candidates(
                    hard_candidates,
                    blocking_candidate_pool.get(id_A, []),
                    id_A,
                    pos_B,
                    mapped_pairs,
                    id2row_B
                )
                for bid in hard_candidates[before:]:
                    candidate_sources[bid] = 1

            scored_hard = []
            for bid in hard_candidates:
                row_B = id2row_B[bid]
                neighs_B = neigh_B.get(bid, {}).get("neighs_attr", [])[:self.args.max_neighbors]
                attr_B = self.row_to_attr_text_B(row_B, override_name=override_name)
                score = self.score_hard_negative(attr_A, attr_B, neighs_A, neighs_B)
                source_rank = candidate_sources.get(bid, 0)
                scored_hard.append((bid, source_rank, score["hard_negative_score"], score))

            scored_hard.sort(key=lambda x: (x[1], x[2]), reverse=True)
            chosen_hard = scored_hard[:self.args.num_hard_neg_per_pos]
            easy_negative_ids = self.sample_easy_negative_ids(
                id_A=id_A,
                pos_B=pos_B,
                mapped_pairs=mapped_pairs,
                id2row_B=id2row_B,
                excluded_ids=hard_candidates
            )

            left_item = self.build_hpc_item(row_A, id_A, "A", neigh_A)
            pos_sample = {
                "left": left_item,
                "right": self.build_hpc_item(row_pos_B, pos_B, "B", neigh_B, override_name=override_name),
                "label": 1
            }
            if not self.args.compact_hpc_json:
                pos_sample["sample_id"] = f"hpc_A{id_A}_B{pos_B}_pos_zh"
                pos_sample["task"] = "hpc"
            samples.append(pos_sample)

            for rank, (bid, _, _, _score_detail) in enumerate(chosen_hard):
                neg_sample = {
                    "left": left_item,
                    "right": self.build_hpc_item(
                        row=id2row_B[bid],
                        entity_id=bid,
                        source="B",
                        neigh_map=neigh_B,
                        override_name=override_name
                    ),
                    "label": 0
                }
                if not self.args.compact_hpc_json:
                    neg_sample["sample_id"] = f"hpc_A{id_A}_B{bid}_neg_{rank}_zh"
                    neg_sample["task"] = "hpc"
                samples.append(neg_sample)

            for rank, bid in enumerate(easy_negative_ids):
                neg_sample = {
                    "left": left_item,
                    "right": self.build_hpc_item(
                        row=id2row_B[bid],
                        entity_id=bid,
                        source="B",
                        neigh_map=neigh_B,
                        override_name=override_name
                    ),
                    "label": 0
                }
                if not self.args.compact_hpc_json:
                    neg_sample["sample_id"] = f"hpc_A{id_A}_B{bid}_easy_neg_{rank}_zh"
                    neg_sample["task"] = "hpc"
                samples.append(neg_sample)

        return samples

    def run(self):
        df_A = self.normalize_schema_columns(pd.read_csv(self.args.dataset_A).fillna(""))
        df_B = self.normalize_schema_columns(pd.read_csv(self.args.dataset_B).fillna(""))
        mapping_df = pd.read_csv(self.args.mapping_csv)

        if self.args.max_samples is not None:
            mapping_df = mapping_df.head(self.args.max_samples)

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

        all_samples = self.build_pair_classification_samples(df_A, df_B, mapping_df, neigh_A, neigh_B)
        random.shuffle(all_samples)
        random.shuffle(all_samples)

        out_path = Path(self.args.output_dir) / "hpc_dataset.jsonl"
        self.save_jsonl(all_samples, out_path)
        print(f"Saved HPC dataset to: {out_path}")
        print(f"Total pair classification samples: {len(all_samples)}")


