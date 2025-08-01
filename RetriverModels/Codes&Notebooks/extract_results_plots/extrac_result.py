import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
import pandas as pd
# ---------- Load JSON ----------
with open("F:\\Fatemeh\\Education\\Master\\Semester2\\NLP\\hw2\\final_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------- Helper ----------
def get_model_name(key):
    if key.startswith("tfidf"):
        return "tfidf"
    elif key.startswith("zeroshot"):
        return "zeroshot"
    elif key.startswith("finetuned"):
        return "finetuned"
    return "unknown"

# ---------- Metrics containers ----------
win_count = defaultdict(int)
rank_sum = defaultdict(int)
rank_count = defaultdict(int)
pairwise = defaultdict(lambda: defaultdict(int))
rank_distribution = defaultdict(lambda: defaultdict(int))

# For significance test (rank matrix)
ranks_matrix = []

for i, entry in enumerate(data):
    answers = [(k, v) for k, v in entry.items() 
           if k.startswith(("tfidf", "zeroshot", "finetuned")) and v.strip()]
    # print(answers)
    # Track ranks
    model_ranks = {"tfidf": [], "zeroshot": [], "finetuned": []}
    for rank, (key, _) in enumerate(answers, start=1):
        model = get_model_name(key)
        rank_sum[model] += rank
        rank_count[model] += 1
        model_ranks[model].append(rank)
        if rank == 1:
            win_count[model] += 1
            # if model == "finetuned":
            #     print(i, entry["question"])

        rank_distribution[rank][model] += 1
    # print(model_ranks)
    # break
    # Convert to average rank per model for this question
    ranks_matrix.append([
        np.mean(model_ranks["tfidf"]),
        np.mean(model_ranks["zeroshot"]),
        np.mean(model_ranks["finetuned"])
    ])
    
    # Pairwise wins
    for i, (key_i, _) in enumerate(answers):
        model_i = get_model_name(key_i)
        for j, (key_j, _) in enumerate(answers):
            if i < j:
                model_j = get_model_name(key_j)
                if model_i != model_j:
                    pairwise[model_i][model_j] += 1

# ---------- Metrics ----------
avg_rank = {m: rank_sum[m] / rank_count[m] for m in rank_sum}
normalized_score = {
    m: (rank_count[m] * len(data) * len(answers) - rank_sum[m]) for m in rank_sum
}

# ---------- Plotting ----------
def plot_bar(metric_dict, title, ylabel):
    models = list(metric_dict.keys())
    values = list(metric_dict.values())
    plt.figure(figsize=(6, 4))
    plt.bar(models, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

df = pd.DataFrame(rank_distribution).T.fillna(0).astype(int)
# Ensure consistent model columns
for col in ["tfidf", "zeroshot", "finetuned"]:
    if col not in df.columns:
        df[col] = 0
# --- Plot ---
df.plot(kind="bar", figsize=(10, 6))
plt.title("Rank Distribution of Models")
plt.xlabel("Rank Position")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.tight_layout()
plt.show()

# plot_bar(win_count, "Win Counts (Rank 1)", "Count")
plot_bar(avg_rank, "Average Rank (Lower is Better)", "Average Rank")
# plot_bar(normalized_score, "Normalized Score", "Score")


# ---------- Statistical Test ----------
ranks_matrix = np.array(ranks_matrix)  # shape = (num_questions, 3 models)
stat, p = friedmanchisquare(ranks_matrix[:, 0], ranks_matrix[:, 1], ranks_matrix[:, 2])
print(f"\nFriedman test: chi2 = {stat:.4f}, p = {p:.4f}")

# Pairwise Wilcoxon tests
models = ["tfidf", "zeroshot", "finetuned"]
for i in range(3):
    for j in range(i+1, 3):
        stat, p = wilcoxon(ranks_matrix[:, i], ranks_matrix[:, j])
        print(f"Wilcoxon {models[i]} vs {models[j]}: stat = {stat:.4f}, p = {p:.4f}")

# ---------- Pairwise Wins Output ----------
print("\n=== Pairwise Wins ===")
for m1 in pairwise:
    for m2, count in pairwise[m1].items():
        print(f"{m1} better than {m2}: {count}")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Pairwise wins (from your result)
df = pd.DataFrame(pairwise).fillna(0)
df = df.reindex(index=df.columns, columns=df.columns).fillna(0)  # Ensure square

# Zero out diagonal (self-comparisons)
for m in df.columns:
    df.loc[m, m] = 0

plt.figure(figsize=(6,5))
sns.heatmap(df.T, annot=True, fmt=".0f", cmap="Blues")
plt.title("Pairwise Wins (rows beat columns)")
plt.show()

print("\n=== Summary ===")
print("Win Counts:", win_count)
print("Average Ranks:", avg_rank)
print("Normalized Scores:", normalized_score)
