import os
import pickle
import tqdm
import numpy as np

from pathlib import Path


USER_DATA_PATH = "./filtered_data/min_freq_160_cleaned_user.pkl"
REPO_EMB_PATH = "./filtered_data/embeddings/repo_embeddings_cpu.pkl"
EMBEDDINGS_DIR = "./filtered_data/embeddings/"
OUTPUT_PATH = os.path.join(EMBEDDINGS_DIR, "user_context_vectors_cpu.pkl")


with open(USER_DATA_PATH, "rb") as f:
    users_data = pickle.load(f)

with open(REPO_EMB_PATH, "rb") as f:
    repo_data = pickle.load(f)


user_history = users_data["user_history"]
repo_embeddings = repo_data["repo_embeddings"]

print(f"Building User Tower for {len(user_history)} users...")

user_context_vectors = {}

for user_id, history in tqdm.tqdm(user_history.items()):
    valid_embs = [repo_embeddings[rid] for rid in history if rid in repo_embeddings]

    if valid_embs:
        # Simple mean pooling
        # TODO: User Exponential weight or last k repos later
        user_vec = np.mean(valid_embs, axis=0)
        user_vec = user_vec / np.linalg.norm(user_vec)  # normalize for cosine
    else:
        user_vec = np.zeros(384, dtype=np.float32)

    user_context_vectors[user_id] = user_vec


with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(
        {
            "user_context_vectors": user_context_vectors,
            "model_name": "all-MiniLM-L6-v2",
            "backend": "cpu",
            "dimension": 384,
            "num_users": len(user_context_vectors),
            "aggregation": "mean_of_all_starred_repos",
        },
        f,
    )

print(f"\nUser Tower saved: {len(user_context_vectors)} user context vectors")
print(f"File: {OUTPUT_PATH}")
