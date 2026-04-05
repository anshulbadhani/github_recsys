"""
Manual evaluation pipeline to debug poor metrics.
"""

from recsys.config import get_config
from recsys.retrieval import FaissRetriever
from recsys.bloom_filter import BloomFilter
from recsys.bandits.neuralucb import NeuralUCB
import numpy as np
import pickle
from random import shuffle

# ============================================================
# SETUP
# ============================================================

config = get_config()
retriever = FaissRetriever(config)
retriever.load_index()

# Load data
with open(config.paths.get_user_history_path(), "rb") as f:
    raw = pickle.load(f)
    
with open(config.paths.get_repo_metadata_path(config.data.min_freq), "rb") as f:
    repos_data = pickle.load(f)

clean_repos = repos_data.get("clean_repos", repos_data)

with open(config.paths.get_repo_embeddings_path(config.model.device), "rb") as f:
    repo_data = pickle.load(f)
repo_embeddings = repo_data["repo_embeddings"]

# ============================================================
# USER & DATA SPLIT
# ============================================================

user_id = 6989
sample_user_history = raw["user_history"][0]  # len: 188

train_set = sample_user_history[:150]  # First 150 repos
test_set = sample_user_history[150:]   # Last 38 repos

print(f"User {user_id}:")
print(f"  Train set: {len(train_set)} repos")
print(f"  Test set: {len(test_set)} repos")

# ============================================================
# CREATE USER EMBEDDING FROM TRAIN SET
# ============================================================

def create_user_embedding(starred_repos, repo_embeddings):
    """Create user embedding by averaging starred repo embeddings."""
    valid_embs = [repo_embeddings[rid] for rid in starred_repos if rid in repo_embeddings]
    
    if valid_embs:
        user_vec = np.mean(valid_embs, axis=0)
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)  # Normalize
    else:
        user_vec = np.zeros(384, dtype=np.float32)
    
    return user_vec

user_emb_train = create_user_embedding(train_set, repo_embeddings)
print(f"  User embedding shape: {user_emb_train.shape}")

# ============================================================
# TRAIN THE BANDIT
# ============================================================

print("\n" + "="*60)
print("TRAINING BANDIT")
print("="*60)

ranker = NeuralUCB(
    dropout_rate=0.2,  # Lower dropout for more stable predictions
    beta=2.0,          # Lower beta for less exploration during eval
    n_uncertainty_samples=10,
    learning_rate=1e-3
)

# Prepare training data
negative_repo_samples = list(set(clean_repos).difference(set(sample_user_history)))

training_data = {}

# Add POSITIVE examples (reward = 1.0)
for repo_id in train_set:
    if repo_id in repo_embeddings:  # Only add if embedding exists
        training_data[int(repo_id)] = 1.0

# Add NEGATIVE examples (reward = 0.0, NOT -1.0!)
for repo_id in negative_repo_samples[:150]:
    if repo_id in repo_embeddings:  # Only add if embedding exists
        training_data[int(repo_id)] = 0.0  # ✅ Use 0.0, not -1.0

# Shuffle for better training
training_list = list(training_data.items())
shuffle(training_list)

print(f"Training on {len(training_list)} examples:")
print(f"  Positive: {sum(1 for _, r in training_list if r == 1.0)}")
print(f"  Negative: {sum(1 for _, r in training_list if r == 0.0)}")

# Train the bandit
for i, (repo_id, reward) in enumerate(training_list):
    item_emb = repo_embeddings[repo_id]
    ranker.update(user_emb_train, item_emb, reward=reward)
    
    if (i + 1) % 50 == 0:
        print(f"  Trained on {i+1}/{len(training_list)} examples...")

print(f"\n✅ Training complete!")
print(f"   Average loss: {ranker.get_average_loss():.4f}")
print(f"   Total updates: {ranker.update_count}")

# Save trained model
ranker.save("./archive/weights.pt")
print(f"   Model saved to ./archive/weights.pt")

# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

print("\n" + "="*60)
print("GENERATING RECOMMENDATIONS")
print("="*60)

# Initialize bloom filter and mark train set as seen
filter = BloomFilter(10000, 0.01)
for repo_id in train_set:
    filter.add(user_id, repo_id)

# FAISS retrieval
_, candidate_ids = retriever.search_by_user(user_id=user_id, k=200)

# Flatten if needed
if candidate_ids.ndim == 2:
    candidate_ids = candidate_ids.flatten()

print(f"FAISS retrieved: {len(candidate_ids)} candidates")

# Bloom filter
novel_candidates = filter.filter_candidates(user_id=user_id, candidate_ids=candidate_ids)
print(f"After Bloom filter: {len(novel_candidates)} novel candidates")

# Prepare embeddings for scoring
repo_embs = []
valid_candidates = []

for repo_id in novel_candidates:
    repo_id = int(repo_id)
    if repo_id in repo_embeddings:
        repo_embs.append(repo_embeddings[repo_id])  # ✅ Just append (384,)
        valid_candidates.append(repo_id)

repo_embs = np.array(repo_embs)  # Shape: (N, 384)
valid_candidates = np.array(valid_candidates)

print(f"Valid candidates for scoring: {len(valid_candidates)}")
print(f"Repo embeddings shape: {repo_embs.shape}")

# Score with NeuralUCB
ucb_scores = ranker.score(user_emb_train, repo_embs)
print(f"UCB scores computed: {len(ucb_scores)}")

# Select top-10
top_10 = ranker.pick_top_k(ucb_scores, valid_candidates, k=10)  # ✅ Fixed method name

print(f"\nTop-10 Recommendations: {top_10}")

# ============================================================
# EVALUATE
# ============================================================

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

test_set_ids = set(test_set)
top_10_set = set(top_10)

# Hit@10
hit_at_10 = 1.0 if len(top_10_set & test_set_ids) > 0 else 0.0

# Precision@10
precision_at_10 = len(top_10_set & test_set_ids) / 10

# Recall@10
recall_at_10 = len(top_10_set & test_set_ids) / len(test_set_ids)

# NDCG@10
dcg = 0.0
for i, repo_id in enumerate(top_10):
    if repo_id in test_set_ids:
        dcg += 1.0 / np.log2(i + 2)

idcg = 0.0
for i in range(min(len(test_set_ids), 10)):
    idcg += 1.0 / np.log2(i + 2)

ndcg_at_10 = dcg / idcg if idcg > 0 else 0.0

print(f"\nResults for User {user_id}:")
print(f"  Test set size: {len(test_set_ids)}")
print(f"  Hit@10:       {hit_at_10:.3f}")
print(f"  Precision@10: {precision_at_10:.3f}")
print(f"  Recall@10:    {recall_at_10:.3f}")
print(f"  NDCG@10:      {ndcg_at_10:.3f}")

# Show which test repos appeared
hits = top_10_set & test_set_ids
if hits:
    print(f"\n✅ Found {len(hits)} test repos in recommendations:")
    for repo_id in hits:
        rank = list(top_10).index(repo_id) + 1
        print(f"     Rank {rank}: Repo {repo_id}")
else:
    print(f"\n❌ No test repos found in top-10!")

print("\n" + "="*60)

"""
Check if train and test repos are semantically different.
"""

import numpy as np

# Compute average embeddings
train_embs = [repo_embeddings[r] for r in train_set if r in repo_embeddings]
test_embs = [repo_embeddings[r] for r in test_set if r in repo_embeddings]

train_avg = np.mean(train_embs, axis=0)
test_avg = np.mean(test_embs, axis=0)

# Normalize
train_avg = train_avg / np.linalg.norm(train_avg)
test_avg = test_avg / np.linalg.norm(test_avg)

# Similarity between train and test centroids
train_test_sim = np.dot(train_avg, test_avg)

print(f"\n{'='*60}")
print("TRAIN vs TEST SIMILARITY")
print(f"{'='*60}")
print(f"Cosine similarity between train and test centroids: {train_test_sim:.4f}")

if train_test_sim < 0.7:
    print("❌ Train and test are DIFFERENT! (similarity < 0.7)")
    print("   This explains why FAISS can't find test repos!")
elif train_test_sim < 0.85:
    print("⚠️  Train and test are SOMEWHAT different (0.7-0.85)")
    print("   Some test repos will be hard to find")
else:
    print("✅ Train and test are SIMILAR (> 0.85)")

# Check individual test repo similarities to train centroid
print(f"\nTest repo similarities to TRAIN embedding:")
test_to_train_sims = []
for repo_id in test_set:
    if repo_id in repo_embeddings:
        repo_emb = repo_embeddings[repo_id]
        sim = np.dot(train_avg, repo_emb) / (np.linalg.norm(repo_emb) + 1e-8)
        test_to_train_sims.append(sim)

print(f"  Mean: {np.mean(test_to_train_sims):.4f}")
print(f"  Median: {np.median(test_to_train_sims):.4f}")
print(f"  Min: {np.min(test_to_train_sims):.4f}")
print(f"  Max: {np.max(test_to_train_sims):.4f}")

# Distribution
below_04 = sum(1 for s in test_to_train_sims if s < 0.4)
print(f"\n  {below_04}/{len(test_to_train_sims)} test repos have similarity < 0.4 to train")
print(f"  These are VERY hard to find with FAISS!")

# 1. Hit@K at higher K values
for k in [10, 20, 50, 100]:
    _, candidates = retriever.search_by_user(user_id, k=k)
    candidates = candidates.flatten()
    overlap = len(set(candidates) & set(test_set))
    print(f"Hit@{k}: {1.0 if overlap > 0 else 0.0} ({overlap} found)")

# 2. Serendipity: Are we finding DIFFERENT repos?
novelty = len(set(top_10) - set(train_set)) / 10
print(f"Novelty: {novelty:.2f} (higher = more novel)")

# 3. Diversity: Are recommendations diverse?
from sklearn.metrics.pairwise import cosine_similarity
top_10_embs = [repo_embeddings[r] for r in top_10 if r in repo_embeddings]
sim_matrix = cosine_similarity(top_10_embs)
avg_sim = (sim_matrix.sum() - len(sim_matrix)) / (len(sim_matrix) * (len(sim_matrix) - 1))
diversity = 1 - avg_sim
print(f"Diversity: {diversity:.2f} (higher = more diverse)")

def create_temporal_user_embedding(starred_repos, repo_embeddings, decay=0.95):
    """Weight recent repos more heavily."""
    valid_embs = []
    weights = []
    
    for i, repo_id in enumerate(starred_repos):
        if repo_id in repo_embeddings:
            valid_embs.append(repo_embeddings[repo_id])
            # Exponential decay: recent repos get higher weight
            weight = decay ** (len(starred_repos) - i - 1)
            weights.append(weight)
    
    if valid_embs:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        user_vec = np.average(valid_embs, axis=0, weights=weights)
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)
    else:
        user_vec = np.zeros(384, dtype=np.float32)
    
    return user_vec

# Use this instead
user_emb_temporal = create_temporal_user_embedding(train_set, repo_embeddings, decay=0.95)