import os
import pickle
import tqdm
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


REPO_DATA_PATH = "./filtered_data/min_freq_160_cleaned_item.pkl"
EMBEDDINGS_DIR = "./filtered_data/embeddings/"

Path(EMBEDDINGS_DIR).mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = os.path.join(EMBEDDINGS_DIR, "repo_embeddings_cpu.pkl")


model = SentenceTransformer("all-MiniLM-L6-v2", device="auto")


with open(REPO_DATA_PATH, "rb") as f:
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


with open(OUTPUT_PATH, "wb") as f:
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
print(f"File: {OUTPUT_PATH}")
