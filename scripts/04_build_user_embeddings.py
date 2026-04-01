import pickle
import tqdm
import numpy as np
from recsys.config import get_config


config = get_config()

with open(config.paths.get_user_history_path(config.data.min_freq), "rb") as f:
    users_data = pickle.load(f)

with open(config.paths.get_repo_metadata_path(config.data.min_freq), "rb") as f:
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


with open(config.paths.get_user_embeddings_path(), "wb") as f:
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
print(f"File: {config.paths.get_user_embeddings_path()}")
