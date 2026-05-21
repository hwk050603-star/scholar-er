from pathlib import Path
import argparse
import csv
import re
import textwrap

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
CASE_STUDY_DIR = PROJECT_ROOT / "case_study"
INPUT_CSV = CASE_STUDY_DIR / "case_neighbor_weights.csv"
CASE_TSV = CASE_STUDY_DIR / "case_study_cases.tsv"
OUTPUT_PREFIX = CASE_STUDY_DIR / "case_neighbor_heatmap"

TARGET_TOPIC_OVERRIDES = {
    "13367": "Agric. Econ.",
    "7293": "Agric. Econ.",
}

NAME_OVERRIDES = {
    "13367": "Li Hua",
    "7293": "Li Hua",
}

AFFILIATION_OVERRIDES = {
    "13367": "GAU",
    "7293": "CAU",
}

NEIGHBOR_TOPIC_OVERRIDES = {
    ("13367", 1): "Rural\nEconomy",
    ("13367", 2): "Urban-Rural\nIntegration",
    ("13367", 3): "Regional\nEconomy",
    ("7293", 1): "Agri-product\nMarket & Trade",
    ("7293", 2): "Agricultural\nEcon. History",
    ("7293", 3): "Agri-product\nMarket & Trade",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw a case-study heatmap from exported SANI neighbor beta weights."
    )
    parser.add_argument("--input", type=str, default=str(INPUT_CSV))
    parser.add_argument("--case_path", type=str, default=str(CASE_TSV))
    parser.add_argument("--output_prefix", type=str, default=str(OUTPUT_PREFIX))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def extract_attr_field(text, field_name):
    marker = f"COL {field_name} VAL "
    start = str(text or "").find(marker)
    if start < 0:
        return ""

    start += len(marker)
    next_positions = []
    for name in ["Name", "Affiliation", "Research Interests", "Papers", "Projects"]:
        next_marker = f" COL {name} VAL "
        pos = text.find(next_marker, start)
        if pos >= 0:
            next_positions.append(pos)

    end = min(next_positions) if next_positions else len(text)
    return text[start:end].strip()


def split_topics(topic_text, max_topics=2):
    parts = re.split(r"[|｜;/；、,，]+", str(topic_text or ""))
    parts = [p.strip() for p in parts if p.strip()]
    return parts[:max_topics]


def wrap_label(label, width=10):
    lines = []
    for line in str(label).splitlines():
        if len(line) <= width:
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=width) or [line])
    return "\n".join(lines[:4])


def load_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_case_texts(path):
    case_map = {}
    path = Path(path)
    if not path.exists():
        return case_map

    with open(path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            parts = [part.strip() for part in line.split("\t")]
            if len(parts) >= 4:
                left_id, right_id, left_text, right_text = parts[:4]
                case_map[(left_id, right_id)] = {
                    "left_text": left_text,
                    "right_text": right_text,
                }
    return case_map


def target_topics(row, side, case_map):
    entity_id = row["left_id"] if side == "left" else row["right_id"]
    if entity_id in TARGET_TOPIC_OVERRIDES:
        return TARGET_TOPIC_OVERRIDES[entity_id]

    key = (row["left_id"], row["right_id"])
    text_key = "left_text" if side == "left" else "right_text"
    target_text = case_map.get(key, {}).get(text_key, "")
    topics = split_topics(extract_attr_field(target_text, "Research Interests"), max_topics=1)
    return topics[0] if topics else "Research interests unavailable"


def target_label(row, side, case_map):
    entity_id = row["left_id"] if side == "left" else row["right_id"]
    name = NAME_OVERRIDES.get(entity_id)
    if name is None:
        name = row["left_name"] if side == "left" else row["right_name"]
    affiliation = AFFILIATION_OVERRIDES.get(entity_id)
    if affiliation:
        return f"{name}\n{affiliation}\n{target_topics(row, side, case_map)}"
    return f"{name}\n{target_topics(row, side, case_map)}"


def neighbor_topic(row):
    key = (row["entity_id"], int(row["neighbor_rank"]))
    if key in NEIGHBOR_TOPIC_OVERRIDES:
        return NEIGHBOR_TOPIC_OVERRIDES[key]

    topics = split_topics(extract_attr_field(row["neighbor_text"], "Research Interests"), max_topics=1)
    return topics[0] if topics else f"Top-{row['neighbor_rank']}"


def build_matrix(rows):
    sides = ["left", "right"]
    max_rank = max(int(row["neighbor_rank"]) for row in rows)
    values = np.zeros((len(sides), max_rank), dtype=float)
    labels = [["" for _ in range(max_rank)] for _ in sides]

    row_by_side = {side: idx for idx, side in enumerate(sides)}
    side_meta = {}
    for row in rows:
        side = row["side"]
        rank = int(row["neighbor_rank"])
        i = row_by_side[side]
        j = rank - 1
        values[i, j] = float(row["beta"])
        labels[i][j] = neighbor_topic(row)
        side_meta[side] = row

    return values, labels, side_meta


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],
        "axes.unicode_minus": False,
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 9.8,
        "ytick.labelsize": 9.8,
        "axes.linewidth": 0.8,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    })


def top_neighbor_label(topic, rank):
    return rf"Neighbor$_i${rank}" + "\n" + topic


def bottom_neighbor_label(topic, rank):
    return rf"Neighbor$_j${rank}" + "\n" + topic


def plot_heatmap(values, labels, side_meta, case_map):
    configure_matplotlib()

    left = side_meta["left"]
    right = side_meta["right"]
    pred = "Match" if str(left["prediction"]) == "1" else "Non-match"

    row_labels = [
        target_label(left, "left", case_map),
        target_label(right, "right", case_map),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=0.7)
    top_positions = {
        tuple(pos)
        for pos in np.argwhere(values >= np.partition(values.ravel(), -2)[-2])
    }

    ax.set_xticks(np.arange(values.shape[1]))
    bottom_labels = [
        bottom_neighbor_label(wrap_label(label, width=18), rank)
        for rank, label in enumerate(labels[1], start=1)
    ]
    ax.set_xticklabels(bottom_labels)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(row_labels)
    for tick_label in ax.get_yticklabels():
        tick_label.set_multialignment("center")
    ax.tick_params(axis="x", bottom=True, labelbottom=True, top=False, labeltop=False, pad=10)
    ax.tick_params(axis="y", pad=2)
    ax.set_xlabel("")
    ax.set_title(f"Neighbor relevance weights: {pred}", pad=24, fontweight="bold")

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(np.arange(values.shape[1]))
    top_labels = [
        top_neighbor_label(wrap_label(label, width=18), rank)
        for rank, label in enumerate(labels[0], start=1)
    ]
    top_ax.set_xticklabels(top_labels)
    top_ax.tick_params(axis="x", length=0, pad=9)
    top_ax.set_xlabel("")

    for col, tick_label in enumerate(top_ax.get_xticklabels()):
        if (0, col) in top_positions:
            tick_label.set_fontweight("bold")
    for col, tick_label in enumerate(ax.get_xticklabels()):
        if (1, col) in top_positions:
            tick_label.set_fontweight("bold")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] >= 0.38 else "black"
            ax.text(
                j,
                i,
                f"{values[i, j]:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(r"Neighbor weight $\beta_i$")

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    output_prefix = Path(args.output_prefix)

    rows = load_rows(args.input)
    case_map = load_case_texts(args.case_path)
    values, labels, side_meta = build_matrix(rows)
    fig = plot_heatmap(values, labels, side_meta, case_map)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")
    fig.savefig(output_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()
