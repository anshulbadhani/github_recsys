import pickle
import pandas as pd

# -----------------------------
# Step 1: Load pickle
# -----------------------------
with open('data/min_freq_40_preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

print("Keys in the pickle:", data.keys())
print(f"Train object attributes: {dir(data['train'])}")  # abbreviated
print(f"Valid object attributes: {dir(data['valid'])[:10]}")
print(f"Test object attributes: {dir(data['test'])[:10]}")

# -----------------------------
# Step 2: Function to convert Interactions to DataFrame
# -----------------------------
def interactions_to_df(interactions):
    df = pd.DataFrame({
        "user_idx": interactions.user_ids,
        "repo_idx": interactions.item_ids
    })
    
    if hasattr(interactions, "ratings") and interactions.ratings is not None:
        df["rating"] = interactions.ratings
    if hasattr(interactions, "timestamps") and interactions.timestamps is not None:
        df["timestamp"] = interactions.timestamps
        
    return df

df_train = interactions_to_df(data["train"])
df_valid = interactions_to_df(data["valid"])
df_test = interactions_to_df(data["test"])

# -----------------------------
# Step 3: Convert list mappings to dicts
# -----------------------------
user_map_dict = {idx: user for idx, user in enumerate(data["user_index2id"])}
repo_map_dict = {idx: repo for idx, repo in enumerate(data["repo_index2id"])}

# -----------------------------
# Step 4: Map indices to real IDs
# -----------------------------
for df in [df_train, df_valid, df_test]:
    df["user"] = df["user_idx"].map(user_map_dict)
    df["repo"] = df["repo_idx"].map(repo_map_dict)

# -----------------------------
# Step 5: Optional - map repo descriptions and languages
# -----------------------------
# repo descriptions
repo_desc_map = data.get("descriptions", {})
# repo languages
repo_lang_map = {idx: data["id2lang"][lang_id] for idx, lang_id in enumerate(data["lang2id"].values())} \
    if "id2lang" in data and "lang2id" in data else {}

for df in [df_train, df_valid, df_test]:
    df["description"] = df["repo"].map(repo_desc_map)
    if repo_lang_map:
        df["language"] = df["repo_idx"].map(repo_lang_map)

# -----------------------------
# Step 6: Drop numerical indices if not needed
# -----------------------------
for df in [df_train, df_valid, df_test]:
    df.drop(columns=["user_idx", "repo_idx"], inplace=True)

# -----------------------------
# Step 7: Save to CSV
# -----------------------------
df_train.to_csv("train.csv", index=False)
df_valid.to_csv("valid.csv", index=False)
df_test.to_csv("test.csv", index=False)

print("Conversion complete! Train, valid, and test CSV files saved.")
print(df_train.head())