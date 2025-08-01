import pandas as pd
import json
import ast

# Load your CSV
df = pd.read_csv("project-29-at-2025-07-31-10-13-23f0d3dd.csv")

# Map number to source column names
number_to_column = {
    1: "tfidf_1",
    2: "tfidf_2",
    3: "tfidf_3",
    4: "zeroshot_4",
    5: "zeroshot_5",
    6: "zeroshot_6",
    7: "finetuned_7",
    8: "finetuned_8",
    9: "finetuned_9"
}

rank_cols = [f"rank{i}" for i in range(1, 10)]
output = []

for _, row in df.iterrows():
    example = {"question": row["question"]}

    for rank_col in rank_cols:
        try:
            number = ast.literal_eval(row[rank_col])[0]["number"]
            source_column = number_to_column[number]
            example[source_column] = row[source_column]
        except Exception as e:
            print(f"Error parsing {rank_col} in row: {row['question']}\n{e}")

    output.append(example)

# Write to JSON
with open("final_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
