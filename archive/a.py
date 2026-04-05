"""
Check if this is a systemic problem or just one weird user.
"""

from recsys.config import get_config
import numpy as np
import pickle

config = get_config()

with open(config.paths.get_user_history_path(), "rb") as f:
    raw = pickle.load(f)
    
with open(config.paths.get_repo_embeddings_path(config.model.device), "rb") as f:
    repo_data = pickle.load(f)
repo_embeddings = repo_data["repo_embeddings"]

# Check multiple users
print("="*60)
print("MULTI-USER DIAGNOSIS")
print("="*60)

train_test_sims = []
mean_test_to_train = []

for i in range(min(20, len(raw["user_history"]))):  # Check first 20 users
    user_history = raw["user_history"][i]
    
    if len(user_history) < 50:  # Skip users with too few stars
        continue
    
    # Split
    train = user_history[:int(0.75 * len(user_history))]
    test = user_history[int(0.75 * len(user_history)):]
    
    # Get embeddings
    train_embs = [repo_embeddings[r] for r in train if r in repo_embeddings]
    test_embs = [repo_embeddings[r] for r in test if r in repo_embeddings]
    
    if len(train_embs) < 10 or len(test_embs) < 5:
        continue
    
    # Compute centroids
    train_avg = np.mean(train_embs, axis=0)
    test_avg = np.mean(test_embs, axis=0)
    
    train_avg = train_avg / (np.linalg.norm(train_avg) + 1e-8)
    test_avg = test_avg / (np.linalg.norm(test_avg) + 1e-8)
    
    # Similarity
    sim = np.dot(train_avg, test_avg)
    train_test_sims.append(sim)
    
    # Individual test repo similarities
    test_sims = [np.dot(train_avg, emb / (np.linalg.norm(emb) + 1e-8)) for emb in test_embs]
    mean_test_to_train.append(np.mean(test_sims))

print(f"\nAnalyzed {len(train_test_sims)} users:\n")

print("Train-Test Centroid Similarity:")
print(f"  Mean: {np.mean(train_test_sims):.4f}")
print(f"  Median: {np.median(train_test_sims):.4f}")
print(f"  Min: {np.min(train_test_sims):.4f}")
print(f"  Max: {np.max(train_test_sims):.4f}")

low_sim_users = sum(1 for s in train_test_sims if s < 0.7)
print(f"\n  {low_sim_users}/{len(train_test_sims)} users have similarity < 0.7")
print(f"  → {low_sim_users/len(train_test_sims)*100:.1f}% of users have VERY DIFFERENT train/test!")

print("\nMean Test Repo Similarity to Train:")
print(f"  Mean: {np.mean(mean_test_to_train):.4f}")
print(f"  Median: {np.median(mean_test_to_train):.4f}")

if np.mean(train_test_sims) < 0.75:
    print("\n❌ SYSTEMIC PROBLEM: Most users have different train/test interests!")
    print("   This is a HARD recommendation problem.")
elif np.mean(train_test_sims) < 0.85:
    print("\n⚠️  MODERATE PROBLEM: Some users have evolving interests.")
else:
    print("\n✅ Users have stable interests - easier problem!")

print("\n" + "="*60)