import pickle
import os
from collections import defaultdict


# Global Variables
DATA_DIR = "./data/"
DATA_FILE = "min_freq_40_preprocessed_data.pkl"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

OUTPUT_DIR = "./filtered_data/"
OUTPUT_FILE = "min_freq_160_clean.pkl"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)


with open(DATA_PATH, "rb") as f:
    raw_data = pickle.load(f)

SPLIT_NAME = "train" # or "test" or "valid"
data = raw_data[SPLIT_NAME]


user_history = defaultdict(list)

for uid, iid in zip(data.user_ids, data.item_ids):
    user_history[uid].append(iid)

print(f"Total users with history: {len(user_history)}")
print(f"Example user {list(user_history.keys())[0]} has {len(user_history[list(user_history.keys())[0]])} repos")


# for embeddings
descriptions = raw_data.get("descriptions", {})
repo_ids_unique = sorted(set(data.item_ids))

print(f"Unique repos: {len(repo_ids_unique)}")

clean_data = {
    "user_history": dict(user_history),
    "descriptions": descriptions,
    "repo_ids_unique": repo_ids_unique,
    "num_users": len(user_history),
    "num_repos": len(repo_ids_unique)
}

# with open(OUTPUT_PATH, "wb") as f:
#     pickle.dump(clean_data, f)

print(f"Clean user history saved to {OUTPUT_PATH}")