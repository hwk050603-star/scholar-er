from pathlib import Path
from functools import lru_cache
import json
import re
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
BLOCKING_DIR = BASE_DIR.parent / "blocking"

input_file = BLOCKING_DIR / "dataset_B_aug.csv"
output_file = DATASET_DIR / "dataset_B_neighbors.json"

top_k = 10
max_source_rows = None   # None means processing all rows

# Original ID column name
id_col = "Id"

# Neighbor filtering thresholds
unit_threshold = 0.7
strong_unit_threshold = 0.92
paper_threshold = 0.20
project_threshold = 0.20
weak_evidence_threshold = 0.12

# Combined-score weights, used only for sorting after filtering
unit_weight = 0.25
paper_weight = 0.45
project_weight = 0.30


# Affiliation inverted-index recall parameters.
# These only affect candidate recall speed and do not change the final neighbor thresholds.
unit_ngram_size = 3
max_unit_bucket_size = 1000


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


def split_by_bar(text: str):
    """
    Split paper/project entries by "|".
    """
    text = safe_text(text)
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def split_paper_titles(text: str):
    """
    Split papers by "|" and pre-extract titles to avoid repeated parsing
    inside the candidate loop.
    """
    return [extract_title_from_paper(paper) for paper in split_by_bar(text)]


def normalize_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, compress whitespace, and normalize
    common Chinese/English punctuation.
    """
    text = safe_text(text).lower()
    text = text.replace("．", ".").replace("。", ".").replace("，", ",").replace("；", ";")
    text = text.replace("（", "(").replace("）", ")").replace("【", "[").replace("】", "]")
    text = text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_unit_for_block(unit: str) -> str:
    unit = normalize_text(unit)
    unit = re.sub(r"[\s,.;:，。；：\-–—_/\\()（）\[\]{}\"'“”‘’]+", "", unit)
    return unit


def unit_block_keys(unit: str):
    """
    Generate recall keys for affiliations.
    B-side internal neighbors mainly come from the same university or department.
    Use prefixes, institution base names, and character n-grams for recall,
    then apply unit_sim as the final filter to avoid exhaustive pairwise comparison.
    """
    unit_norm = normalize_unit_for_block(unit)
    if not unit_norm:
        return set()

    keys = {f"unit:{unit_norm}"}
    for size in (4, 6, 8, 10):
        if len(unit_norm) >= size:
            keys.add(f"prefix{size}:{unit_norm[:size]}")

    base_ends = ["大学", "研究院", "研究所", "学院", "中心", "实验室"]
    base_candidates = []
    for suffix in base_ends:
        pos = unit_norm.find(suffix)
        if pos >= 2:
            base_candidates.append(unit_norm[:pos + len(suffix)])
    if base_candidates:
        keys.add(f"base:{min(base_candidates, key=len)}")

    if len(unit_norm) <= unit_ngram_size:
        keys.add(f"gram:{unit_norm}")
    else:
        for i in range(len(unit_norm) - unit_ngram_size + 1):
            keys.add(f"gram:{unit_norm[i:i + unit_ngram_size]}")
    return keys


@lru_cache(maxsize=300000)
def levenshtein_distance(text1: str, text2: str) -> int:
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
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 1.0

    dist = levenshtein_distance(text1, text2)
    return 1 - dist / max_len

def strip_noise_from_paper(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    text = re.sub(r"https?://\S+|doi\s*[:：]?\s*\S+", " ", text)
    text = re.sub(r"\[\[?\s*[jcp]\s*\]?\]?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(sci|ssci|ei|cssci|ccf-[abc]|jcr\s*q[1-4]|top)\b", " ", text)
    text = re.sub(r"\([^)]{0,40}(收录|检索|影响因子|if\s*=|sci|ei|cssci|jcr|ccf)[^)]{0,40}\)", " ", text)
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    text = re.sub(r"\b\d+\s*[(（]?\d*[)）]?\s*[:：]\s*\d+[-–—]\d+\b", " ", text)
    text = re.sub(r"\b(vol\.?|volume|no\.?|issue|pp\.?|pages?)\s*[\d\-–—:,.() ]+", " ", text)
    text = re.sub(r"[\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;，。；")


def remove_author_prefix(text: str) -> str:
    """
    B-side paper formats are mixed: English records often use
    "authors. title. venue", while Chinese records often use author/title/venue patterns.
    Use lightweight rules to remove obvious author prefixes; keep the original
    text when uncertain.
    """
    text = strip_noise_from_paper(text)
    if not text:
        return ""

    separators = [".", "。"]
    for sep in separators:
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        if len(parts) >= 2:
            first = parts[0]
            second = parts[1]
            first_has_author_signal = (
                "," in first or ";" in first or " et al" in first or
                bool(re.search(r"[\u4e00-\u9fff]{2,4}[,，、]", first)) or
                bool(re.search(r"\b[a-z]\.?\s*[a-z][a-z\-]+", first))
            )
            if first_has_author_signal and len(second) >= 6:
                return second

    # Also remove the prefix before a comma when it clearly looks like an author list.
    comma_parts = re.split(r"[,，]", text, maxsplit=1)
    if len(comma_parts) == 2 and len(comma_parts[1].strip()) >= 8:
        first = comma_parts[0]
        if re.search(r"\*|#| et al|[\u4e00-\u9fff]{2,4}$", first):
            return comma_parts[1].strip()

    return text


def extract_title_from_paper(paper_text: str) -> str:
    """
    Coarse title/topic-fragment extraction.
    Prefer removing author and venue noise; fall back to the cleaned full paper text.
    """
    text = remove_author_prefix(paper_text)
    if not text:
        return ""
    text = re.split(
        r"\b(journal|transactions|proceedings|conference|letters|science|nature|acta|学报|期刊|研究|journal)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0].strip(" .,:;，。；")
    return text or strip_noise_from_paper(paper_text)


def char_ngrams(text: str, n: int = 3):
    text = normalize_text(text)
    text = re.sub(r"[\s,.;:，。；：\-–—_/\\()（）\[\]{}\"'“”‘’]+", "", text)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def token_set(text: str):
    text = normalize_text(text)
    toks = re.split(r"[\s,.;:，。；：\-–—_/\\()（）\[\]{}\"'“”‘’]+", text)
    return {t for t in toks if len(t) >= 2}


def jaccard_similarity(s1, s2) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def hybrid_text_similarity(text1: str, text2: str) -> float:
    """
    Combine edit distance, token overlap, and character 3-gram overlap.
    Levenshtein works well for short titles, while n-gram overlap is more robust
    for Chinese titles and formatting noise.
    """
    text1 = strip_noise_from_paper(text1)
    text2 = strip_noise_from_paper(text2)
    if not text1 or not text2:
        return 0.0
    lev_sim = levenshtein_similarity(text1, text2)
    tok_sim = jaccard_similarity(token_set(text1), token_set(text2))
    char_sim = jaccard_similarity(char_ngrams(text1), char_ngrams(text2))
    return 0.50 * lev_sim + 0.25 * tok_sim + 0.25 * char_sim


def single_paper_similarity(paper_text1: str, paper_text2: str) -> float:
    """
    Single-paper similarity: take the larger score between title/topic-fragment
    similarity and cleaned full-text similarity.
    """
    title1 = extract_title_from_paper(paper_text1)
    title2 = extract_title_from_paper(paper_text2)
    title_sim = hybrid_text_similarity(title1, title2)
    full_sim = hybrid_text_similarity(paper_text1, paper_text2)
    return max(title_sim, full_sim)


def single_paper_title_similarity(title1: str, title2: str) -> float:
    """
    Single-paper similarity after title extraction.
    """
    return hybrid_text_similarity(title1, title2)


def multi_paper_similarity(papers1, papers2) -> float:
    """
    For each paper in papers1, find the most similar paper in papers2 and average
    those best scores.
    """
    if not papers1 and not papers2:
        return 1.0
    if not papers1 or not papers2:
        return 0.0

    best_scores = []
    for p1 in papers1:
        best = 0.0
        for p2 in papers2:
            sim = single_paper_title_similarity(p1, p2)
            if sim > best:
                best = sim
        best_scores.append(best)

    return sum(best_scores) / len(best_scores)


def single_project_similarity(project_text1: str, project_text2: str) -> float:
    """
    Single-project similarity: project names also contain many IDs and time noise,
    so use hybrid text similarity.
    """
    return hybrid_text_similarity(project_text1, project_text2)


def multi_project_similarity(projects1, projects2) -> float:
    """
    For each project in projects1, find the most similar project in projects2
    and average those best scores.
    """
    if not projects1 and not projects2:
        return 1.0
    if not projects1 or not projects2:
        return 0.0

    best_scores = []
    for pr1 in projects1:
        best = 0.0
        for pr2 in projects2:
            sim = single_project_similarity(pr1, pr2)
            if sim > best:
                best = sim
        best_scores.append(best)

    return sum(best_scores) / len(best_scores)


def build_attr(row) -> str:
    name = safe_text(row["Name"])
    unit = safe_text(row["Affiliation"])
    paper = safe_text(row["Papers"])
    project = safe_text(row["Projects"])

    attr = (
        f"COL Name VAL {name} "
        f"COL Affiliation VAL {unit} "
        f"COL Papers VAL {paper} "
        f"COL Projects VAL {project}"
    )
    return attr


df = pd.read_csv(input_file)

if max_source_rows is not None:
    source_df = df.head(max_source_rows)
else:
    source_df = df

print(f"Start processing {len(source_df)} source records from {len(df)} total candidates.")


records = []
for idx, row in df.iterrows():
    unit = safe_text(row["Affiliation"])
    records.append({
        "idx": idx,
        "id": normalize_id(row[id_col]),
        "unit": unit,
        "unit_keys": unit_block_keys(unit),
        "papers": split_paper_titles(row["Papers"]),
        "projects": split_by_bar(row["Projects"]),
        "attr": build_attr(row)
    })

unit_inverted_index = {}
for record_pos, record in enumerate(records):
    for key in record["unit_keys"]:
        unit_inverted_index.setdefault(key, []).append(record_pos)

unit_inverted_index = {
    key: values
    for key, values in unit_inverted_index.items()
    if len(values) <= max_unit_bucket_size
}

avg_bucket_hits = (
    sum(len(v) for v in unit_inverted_index.values()) / len(unit_inverted_index)
    if unit_inverted_index else 0
)
print(
    f"Affiliation inverted index: keys={len(unit_inverted_index)}, "
    f"avg_bucket_size={avg_bucket_hits:.2f}"
)

if max_source_rows is not None:
    source_records = records[:max_source_rows]
else:
    source_records = records


results = []

for current in tqdm(source_records, total=len(source_records), desc="Matching neighbors"):
    current_id = current["id"]
    current_unit = current["unit"]
    current_papers = current["papers"]
    current_projects = current["projects"]
    current_attr = current["attr"]

    neighbors_scored = []
    candidate_positions = set()
    for key in current["unit_keys"]:
        candidate_positions.update(unit_inverted_index.get(key, []))

    for candidate_pos in candidate_positions:
        candidate = records[candidate_pos]
        if current["idx"] == candidate["idx"]:
            continue

        candidate_id = candidate["id"]
        candidate_unit = candidate["unit"]
        candidate_papers = candidate["papers"]
        candidate_projects = candidate["projects"]

        if not current_unit or not candidate_unit:
            continue

        has_paper_evidence = bool(current_papers and candidate_papers)
        has_project_evidence = bool(current_projects and candidate_projects)
        if not has_paper_evidence and not has_project_evidence:
            continue

        # Use affiliation similarity as a fast filter before computing long paper/project similarities.
        unit_sim = levenshtein_similarity(current_unit, candidate_unit)
        if unit_sim <= unit_threshold:
            continue

        paper_sim = (
            multi_paper_similarity(current_papers, candidate_papers)
            if has_paper_evidence else 0.0
        )
        project_sim = (
            multi_project_similarity(current_projects, candidate_projects)
            if has_project_evidence else 0.0
        )

        # Neighbor decision criteria
        is_neighbor = (
            (has_paper_evidence and paper_sim >= paper_threshold) or
            (has_project_evidence and project_sim >= project_threshold) or
            (
                unit_sim >= strong_unit_threshold and
                max(paper_sim, project_sim) >= weak_evidence_threshold
            )
        )

        if not is_neighbor:
            continue

        # Combined score, used only for sorting
        score = (
            unit_weight * unit_sim +
            paper_weight * paper_sim +
            project_weight * project_sim
        )

        neighbors_scored.append({
            "neighbor_id": candidate_id,
            "score": round(score, 6),
            "unit_sim": round(unit_sim, 6),
            "paper_sim": round(paper_sim, 6),
            "project_sim": round(project_sim, 6),
            "neighbor_attr": candidate["attr"]
        })

    # Sort by combined score in descending order and keep top-k neighbors
    neighbors_scored.sort(key=lambda x: x["score"], reverse=True)
    top_neighbors = neighbors_scored[:top_k]

    neighs = [item["neighbor_id"] for item in top_neighbors]
    neighs_attr = [item["neighbor_attr"] for item in top_neighbors]

    results.append({
        "b_id": current_id,
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

print(f"\nB neighbor search finished. Results saved to: {output_file}")
print(
    f"B neighbor stats: total={len(results)}, "
    f"top_k={top_k}, "
    f"full_top_k={full_top_k_count} ({full_top_k_count / len(results) * 100 if results else 0.0:.2f}%), "
    f"zero_neighbors={zero_neighbor_count}, "
    f"avg_neighbors={avg_neighbor_count:.2f}"
)
print(f"B neighbor-count distribution: {count_distribution}")
