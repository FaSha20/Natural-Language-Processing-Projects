import json

# Load your original JSON data
with open(r"F:\Fatemeh\Education\Master\Semester2\NLP\hw1\scintists_data (3).json", "r", encoding="utf-8") as f:
    records = json.load(f)

# # Wrap each item under "data" key
# wrapped = [{"data": record} for record in records]

# # Save it to a new file
# with open("labelstudio_formatted.json", "w", encoding="utf-8") as f:
#     json.dump(wrapped, f, ensure_ascii=False, indent=2)

# print("✅ Done! Data is ready for Label Studio.")


import json

def count_words_in_value(value):
    if isinstance(value, str):
        return len(value.split())
    elif isinstance(value, list):
        return sum(count_words_in_value(item) for item in value)
    elif isinstance(value, dict):
        return sum(count_words_in_value(v) for v in value.values())
    else:
        return 0

def calculate_mean_word_count(data):

    total_words = 0
    record_count = 0

    for record in data[:1]:
        word_count = count_words_in_value(record)
        total_words += word_count
        record_count += 1

    if record_count == 0:
        return 0
    return total_words / record_count


mean_words = calculate_mean_word_count(records)
print(f"Mean number of words per record: {mean_words:.2f}")
