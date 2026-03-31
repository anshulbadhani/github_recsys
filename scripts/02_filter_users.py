"""
Step 2: Filter and clean user interaction histories.
 
This script processes raw preprocessed data and extracts user interaction
histories (which repositories each user starred).
"""
import sys
import pickle
from pathlib import Path
from collections import defaultdict


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config

config = get_config()
with open(config.paths.get_raw_data_path(config.data.min_freq), "rb") as f:
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

with open(config.paths.get_user_history_path(config.data.min_freq), "wb") as f:
    pickle.dump(clean_data, f)

print(f"Clean user history saved to {config.paths.get_user_history_path()}")