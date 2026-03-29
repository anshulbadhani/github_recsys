# import pickle
# from spotlight import Interactions, SequenceInteractions

# with open("data/min_freq_160_preprocessed_data.pkl", "rb") as f:
#     raw = pickle.load(f)


# # user_ids = raw["train"].user_ids
# # item_ids = raw["train"].item_ids
# # ratings  = raw["train"].ratings
# # interactions = Interactions(user_ids, item_ids, ratings)
# interactions = raw["train"] # already of type interactions
# interactions.to_sequence(10, 5, 1)
# print(interactions.sequences.sequences.shape)
# # if hasattr(interactions.sequences, 'targets') and interactions.sequences.targets is not None:
# #     print("sequences.targets shape:", interactions.sequences.targets.shape)

# print(interactions.test_sequences.sequences[0:2])

# # print("Keys:", list(raw.keys()) if isinstance(raw, dict) else "Not a dict")
# # print("Example:", {k: type(v) for k, v in raw.items()} if isinstance(raw, dict) else "raw_data type")
# # print(raw["train"].__dict__.sequences)


# AI GENERATED #
import pickle
import numpy as np
from collections import Counter
import os

# ====================== CONFIG ======================
USER_EMB_PATH = "./filtered_data/embeddings/user_context_vectors_cpu.pkl"
REPO_EMB_PATH = "./filtered_data/embeddings/repo_embeddings_cpu.pkl"
USER_DATA_PATH = (
    "./filtered_data/min_freq_160_cleaned_user.pkl"  # the one with user_history
)

print("🔍 Loading embeddings and data...")

# Load User Tower
with open(USER_EMB_PATH, "rb") as f:
    user_data = pickle.load(f)

# Load Repo Tower
with open(REPO_EMB_PATH, "rb") as f:
    repo_data = pickle.load(f)

# Load user history (for validation)
with open(USER_DATA_PATH, "rb") as f:
    user_raw = pickle.load(f)

user_vectors = user_data["user_context_vectors"]
repo_vectors = repo_data["repo_embeddings"]
user_history = user_raw["user_history"]

print(f"✅ Loaded {len(user_vectors)} user vectors")
print(f"✅ Loaded {len(repo_vectors)} repo vectors")
print(f"Dimension: {user_vectors[list(user_vectors.keys())[0]].shape[0]}")

# ====================== BASIC SANITY CHECKS ======================

print("\n" + "=" * 60)
print("RUNNING SANITY TESTS")
print("=" * 60)

# 1. Shape and type check
sample_user_id = list(user_vectors.keys())[0]
sample_repo_id = list(repo_vectors.keys())[0]

user_vec = user_vectors[sample_user_id]
repo_vec = repo_vectors[sample_repo_id]

print(f"1. Vector shapes → User: {user_vec.shape}, Repo: {repo_vec.shape}")
assert (
    user_vec.shape == repo_vec.shape
), "❌ Dimension mismatch between user and repo vectors!"
assert user_vec.dtype == repo_vec.dtype, "❌ Dtype mismatch!"

# 2. Normalization check (very important for cosine similarity)
user_norm = np.linalg.norm(user_vec)
repo_norm = np.linalg.norm(repo_vec)

print(
    f"2. Normalization check → User norm: {user_norm:.4f}, Repo norm: {repo_norm:.4f}"
)
assert abs(user_norm - 1.0) < 1e-5, f"❌ User vector not normalized! (norm={user_norm})"
assert abs(repo_norm - 1.0) < 1e-5, f"❌ Repo vector not normalized! (norm={repo_norm})"

# 3. Check if user history repos actually exist in repo embeddings
print("3. Checking user history consistency...")
consistent = 0
total = 0

for uid, history in list(user_history.items())[:100]:  # check first 100 users
    for rid in history:
        total += 1
        if rid in repo_vectors:
            consistent += 1

print(
    f"   {consistent}/{total} history items have embeddings ({consistent/total*100:.1f}%)"
)

# 4. Quick similarity test - Pick one user and find top-5 similar repos
print("\n4. Quick recommendation test for one user...")

test_user_id = sample_user_id
user_vec = user_vectors[test_user_id]

# Compute cosine similarity with all repos (fast enough for testing)
similarities = []
for rid, rvec in repo_vectors.items():
    sim = np.dot(user_vec, rvec)  # since both are normalized → cosine similarity
    similarities.append((rid, sim))

# Sort by similarity
similarities.sort(key=lambda x: x[1], reverse=True)

print(f"\n✅ Top 10 recommended repos for user {test_user_id}:")
for i, (rid, score) in enumerate(similarities[:10], 1):
    print(f"  {i:2d}. Repo {rid} | Score: {score:.4f}")

# 5. Check if user's own starred repos appear high in recommendations (good signal)
print(f"\n5. Checking overlap with user's actual history...")
user_starred = set(user_history.get(test_user_id, []))

top_50_repos = [rid for rid, _ in similarities[:50]]
overlap = len(user_starred & set(top_50_repos))

print(
    f"   User's starred repos in top-50 recommendations: {overlap}/{len(user_starred)}"
)

if overlap > 0:
    print("   ✅ Good signal: Some of user's starred repos rank highly!")
else:
    print("   ⚠️  Warning: No overlap in top-50 — might need better user modeling")

print("\n" + "=" * 60)
print(
    "🎉 ALL BASIC TESTS PASSED!"
    if overlap >= 1
    else "⚠️ TESTS COMPLETED (check warning above)"
)
print("=" * 60)
