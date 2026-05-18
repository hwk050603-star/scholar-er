from pathlib import Path
from functools import lru_cache
import json
import re
import pandas as pd
from tqdm import tqdm  # Progress bar

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
BLOCKING_DIR = BASE_DIR.parent / "blocking"

input_file = BLOCKING_DIR / "dataset_A_aug.csv"
output_file = DATASET_DIR / "dataset_A_neighbors.json"

top_k = 10
unit_threshold = 0.7
unit_weight = 0.4
field_weight = 0.6
max_source_rows = None   # None means processing all rows

# Original ID column name
id_col = "Id"


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


@lru_cache(maxsize=200000)
def levenshtein_distance(text1: str, text2: str) -> int:
    """Compute the Levenshtein edit distance."""
    if text1 == text2:
        return 0
    if not text1:
        return len(text2)
    if not text2:
        return len(text1)

    if len(text1) < len(text2):
        text1, text2 = text2, text1

    previous_row = list(range(len(text2) + 1))
    for i, char1 in enumerate(text1, start=1):
        current_row = [i]
        for j, char2 in enumerate(text2, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            replace_cost = previous_row[j - 1] + (char1 != char2)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row

    return previous_row[-1]


def levenshtein_similarity(text1: str, text2: str) -> float:
    """
    Normalized Levenshtein similarity:
    sim = 1 - distance / max(len(text1), len(text2))
    Return a value in [0, 1].
    """
    text1 = safe_text(text1)
    text2 = safe_text(text2)

    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0

    dist = levenshtein_distance(text1, text2)
    return 1 - dist / max_len


def split_fields(field_text: str):
    """
    Split research interests into labels with common separators.
    """
    field_text = safe_text(field_text)
    if not field_text:
        return []

    parts = re.split(r"[|、，,;；/]+", field_text)
    parts = [p.strip().lower() for p in parts if p.strip()]
    return parts


def field_similarity(fields1, fields2) -> float:
    """
    Research-interest similarity.
    For each label in fields1, find its maximum similarity against fields2,
    then average those maximum scores.

    Formula:
    sim(A, B) = average( max(sim(a, b)) for a in A, b in B )

    Return a value in [0, 1].
    """
    if not fields1 and not fields2:
        return 1.0
    if not fields1 or not fields2:
        return 0.0

    best_scores = []
    for f1 in fields1:
        best = 0.0
        for f2 in fields2:
            sim = levenshtein_similarity(f1, f2)
            if sim > best:
                best = sim
        best_scores.append(best)

    return sum(best_scores) / len(best_scores)


def build_attr(row) -> str:
    name = safe_text(row["Name"])
    unit = safe_text(row["Affiliation"])
    field = safe_text(row["Research Interests"])
    paper = safe_text(row["Papers"])
    project = safe_text(row["Projects"])

    attr = (
        f"COL Name VAL {name} "
        f"COL Affiliation VAL {unit} "
        f"COL Research Interests VAL {field} "
        f"COL Papers VAL {paper} "
        f"COL Projects VAL {project}"
    )
    return attr

df = pd.read_csv(input_file)

if max_source_rows is not None:
    source_df = df.head(max_source_rows)
else:
    source_df = df


records = []
for idx, row in df.iterrows():
    records.append({
        "idx": idx,
        "id": normalize_id(row[id_col]),
        "unit": safe_text(row["Affiliation"]),
        "fields": split_fields(row["Research Interests"]),
        "attr": build_attr(row)
    })

if max_source_rows is not None:
    source_records = records[:max_source_rows]
else:
    source_records = records


results = []

for current in tqdm(source_records, total=len(source_records), desc="Matching neighbors"):
    current_id = current["id"]
    current_unit = current["unit"]
    current_fields = current["fields"]

    current_attr = current["attr"]
    neighbors_scored = []

    for candidate in records:
        if current["idx"] == candidate["idx"]:
            continue

        candidate_id = candidate["id"]
        candidate_unit = candidate["unit"]
        candidate_fields = candidate["fields"]

        if not current_unit or not current_fields or not candidate_unit or not candidate_fields:
            continue

        # 1. Affiliation similarity
        unit_sim = levenshtein_similarity(current_unit, candidate_unit)
        if unit_sim < unit_threshold:
            continue

        # 2. Research-interest similarity: label-level fuzzy matching with average maximum scores
        field_sim = field_similarity(current_fields, candidate_fields)

        # 3. Only candidates above the affiliation threshold are considered
        score = unit_weight * unit_sim + field_weight * field_sim
        neighbors_scored.append((score, candidate_id, candidate["attr"]))

    # Sort by combined score and keep top-k neighbors
    neighbors_scored.sort(key=lambda x: x[0], reverse=True)
    top_neighbors = neighbors_scored[:top_k]

    neighs = []
    neighs_attr = []

    for _, neighbor_id, neighbor_attr in top_neighbors:
        neighs.append(neighbor_id)
        neighs_attr.append(neighbor_attr)

    results.append({
        "a_id": current_id,
        "attr": current_attr,
        "neighs": neighs,
        "neighs_attr": neighs_attr
    })


neighbor_counts = [len(item["neighs"]) for item in results]
full_top_k_count = sum(1 for count in neighbor_counts if count >= top_k)
zero_neighbor_count = sum(1 for count in neighbor_counts if count == 0)
avg_neighbor_count = sum(neighbor_counts) / len(neighbor_counts) if neighbor_counts else 0.0
count_distribution = {
    count: neighbor_counts.count(count)
    for count in sorted(set(neighbor_counts))
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\nNeighbor search finished. Results saved to: {output_file}")
print(
    f"A neighbor stats: total={len(results)}, "
    f"top_k={top_k}, "
    f"full_top_k={full_top_k_count} ({full_top_k_count / len(results) * 100 if results else 0.0:.2f}%), "
    f"zero_neighbors={zero_neighbor_count}, "
    f"avg_neighbors={avg_neighbor_count:.2f}"
)
print(f"A neighbor-count distribution: {count_distribution}")
