"""
Step 3: Build repository embeddings (Repo Tower).

This script generates dense vector embeddings for each repository using
a sentence transformer model on the repo name, description, and language.
"""

import pickle
import tqdm
from sentence_transformers import SentenceTransformer
from recsys.config import get_config

config = get_config()
model = SentenceTransformer(
    config.model.embedding_model_name, device=config.model.device
)

with open(config.paths.get_repo_metadata_path(config.data.min_freq), "rb") as f:
    repos_data = pickle.load(f)

clean_repos = repos_data.get("clean_repos", repos_data)
print(f"Processing {len(clean_repos)} repositories...")

repo_embeddings = {}

for repo_id, info in tqdm.tqdm(clean_repos.items()):
    prompt = (
        info.get("prompt")
        or f"{info.get('repo_name', repo_id)} | {info.get('description', 'No description')} | {info.get('language', 'Unknown')}"
    )

    embedding = model.encode(
        prompt,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
    )
    repo_embeddings[repo_id] = embedding


with open(config.paths.get_repo_embeddings_path(), "wb") as f:
    pickle.dump(
        {
            "repo_embeddings": repo_embeddings,
            "model_name": "all-MiniLM-L6-v2",
            "backend": "cpu",
            "dimension": 384,
            "num_repos": len(repo_embeddings),
        },
        f,
    )

print(f"\nRepo Tower saved: {len(repo_embeddings)} embeddings")
print(f"File: {config.paths.get_repo_embeddings_path()}")
