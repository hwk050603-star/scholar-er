from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURE_DIR = PROJECT_ROOT / "figure"

k_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
f1_scores = np.array([91.63, 93.05, 93.04, 93.40, 92.49, 92.21, 91.45, 91.80, 91.77])
p_scores = np.array([92.96, 93.15, 93.32, 92.84, 92.03, 92.68, 89.42, 91.25, 91.58])
r_scores = np.array([90.34, 92.96, 92.76, 93.96, 92.96, 91.75, 93.56, 92.35, 91.95])

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
})

fig, ax = plt.subplots(figsize=(5.2, 3.6))

ax.plot(
    k_values, f1_scores,
    marker="o",
    markersize=4,
    linewidth=1.6,
    color="#4C72B0",
    markeredgecolor="white",
    label="F1"
)

ax.plot(
    k_values, p_scores,
    marker="x",
    markersize=4,
    linewidth=1.6,
    linestyle="--",
    color="#DD8452",
    label="P"
)

ax.plot(
    k_values, r_scores,
    marker="s",
    markersize=3.5,
    linewidth=1.6,
    linestyle=":",
    color="#55A868",
    markeredgecolor="white",
    label="R"
)

best_idx = np.nanargmax(f1_scores)
best_k = k_values[best_idx]
best_f1 = f1_scores[best_idx]
ax.scatter(
    best_k,
    best_f1,
    s=42,
    color="#4C72B0",
    edgecolor="black",
    linewidth=0.8,
    zorder=6
)

ax.set_xlabel("Neighborhood size K")
ax.set_ylabel("Score")
ax.set_title("Impact of Neighborhood Size K")

ax.set_xticks(k_values)
ax.set_ylim(88, 95)
ax.set_yticks(np.arange(88, 96, 1))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=True,
    fancybox=False,
    edgecolor="#CCCCCC",
    loc="lower right",
    handlelength=1.6,
    borderpad=0.35,
    labelspacing=0.25,
    handletextpad=0.55,
    markerscale=0.8,
)

fig.tight_layout()

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGURE_DIR / "k_analysis_prf.png", dpi=300, bbox_inches="tight")
fig.savefig(FIGURE_DIR / "k_analysis_prf.pdf", bbox_inches="tight")
plt.close()
