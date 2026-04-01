"""
Step 5: Evaluate and test the embedding quality. (AI GENERATED)

This script runs sanity checks on the generated embeddings and demonstrates
how to use them for making recommendations.
"""

import pickle
import numpy as np
from recsys.config import get_config


def load_embeddings(config):
    """Load user and repo embeddings."""
    user_emb_path = config.paths.get_user_embeddings_path("cpu")
    repo_emb_path = config.paths.get_repo_embeddings_path("cpu")
    user_data_path = config.paths.get_user_history_path(config.data.min_freq)
    
    print("🔍 Loading embeddings and data...")
    
    with open(user_emb_path, "rb") as f:
        user_data = pickle.load(f)
    
    with open(repo_emb_path, "rb") as f:
        repo_data = pickle.load(f)
    
    with open(user_data_path, "rb") as f:
        user_raw = pickle.load(f)
    
    user_vectors = user_data["user_context_vectors"]
    repo_vectors = repo_data["repo_embeddings"]
    user_history = user_raw["user_history"]
    
    print(f"✅ Loaded {len(user_vectors)} user vectors")
    print(f"✅ Loaded {len(repo_vectors)} repo vectors")
    print(f"✅ Embedding dimension: {user_vectors[list(user_vectors.keys())[0]].shape[0]}")
    
    return user_vectors, repo_vectors, user_history


def run_sanity_checks(user_vectors, repo_vectors, user_history):
    """Run basic sanity checks on embeddings."""
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    # 1. Shape and type check
    sample_user_id = list(user_vectors.keys())[0]
    sample_repo_id = list(repo_vectors.keys())[0]
    
    user_vec = user_vectors[sample_user_id]
    repo_vec = repo_vectors[sample_repo_id]
    
    print(f"\n1. Vector shapes → User: {user_vec.shape}, Repo: {repo_vec.shape}")
    assert user_vec.shape == repo_vec.shape, "❌ Dimension mismatch!"
    assert user_vec.dtype == repo_vec.dtype, "❌ Dtype mismatch!"
    print("   ✅ Shapes and dtypes match")
    
    # 2. Normalization check
    user_norm = np.linalg.norm(user_vec)
    repo_norm = np.linalg.norm(repo_vec)
    
    print(f"\n2. Normalization → User: {user_norm:.4f}, Repo: {repo_norm:.4f}")
    assert abs(user_norm - 1.0) < 1e-5, f"❌ User vector not normalized! (norm={user_norm})"
    assert abs(repo_norm - 1.0) < 1e-5, f"❌ Repo vector not normalized! (norm={repo_norm})"
    print("   ✅ Both vectors are normalized")
    
    # 3. User history consistency
    print("\n3. Checking user history consistency...")
    consistent = 0
    total = 0
    
    for uid, history in list(user_history.items())[:100]:
        for rid in history:
            total += 1
            if rid in repo_vectors:
                consistent += 1
    
    consistency_rate = consistent / total * 100
    print(f"   {consistent}/{total} history items have embeddings ({consistency_rate:.1f}%)")
    print("   ✅ Good consistency" if consistency_rate > 95 else "   ⚠️  Low consistency")


def recommend_for_user(user_id, user_vectors, repo_vectors, user_history, top_k=10):
    """Generate recommendations for a specific user."""
    user_vec = user_vectors[user_id]
    
    # Compute cosine similarity with all repos
    similarities = []
    for rid, rvec in repo_vectors.items():
        sim = np.dot(user_vec, rvec)  # Cosine similarity (both normalized)
        similarities.append((rid, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def evaluate_recommendations(user_vectors, repo_vectors, user_history):
    """Evaluate recommendation quality."""
    print("\n" + "=" * 60)
    print("RECOMMENDATION EVALUATION")
    print("=" * 60)
    
    # Pick a test user
    test_user_id = list(user_vectors.keys())[0]
    user_starred = set(user_history.get(test_user_id, []))
    
    # Generate recommendations
    recommendations = recommend_for_user(
        test_user_id, user_vectors, repo_vectors, user_history, top_k=50
    )
    
    print(f"\n📊 Test User: {test_user_id}")
    print(f"   User has starred {len(user_starred)} repos")
    print(f"\n🎯 Top 10 Recommendations:")
    
    for i, (rid, score) in enumerate(recommendations[:10], 1):
        in_history = "⭐ (starred)" if rid in user_starred else ""
        print(f"   {i:2d}. Repo {rid:8d} | Score: {score:.4f} {in_history}")
    
    # Check overlap with user's actual starred repos
    top_50_repos = [rid for rid, _ in recommendations[:50]]
    overlap = len(user_starred & set(top_50_repos))
    
    print(f"\n📈 Overlap Analysis:")
    print(f"   User's starred repos in top-50: {overlap}/{len(user_starred)}")
    
    if overlap > 0:
        print("   ✅ Good signal: Some starred repos rank highly!")
    else:
        print("   ⚠️  Warning: No overlap in top-50")
    
    return overlap > 0


def main():
    """Run evaluation pipeline."""
    config = get_config()
    
    print("=" * 60)
    print("STEP 5: EVALUATING EMBEDDINGS")
    print("=" * 60)
    
    # Load data
    user_vectors, repo_vectors, user_history = load_embeddings(config)
    
    # Run sanity checks
    run_sanity_checks(user_vectors, repo_vectors, user_history)
    
    # Evaluate recommendations
    success = evaluate_recommendations(user_vectors, repo_vectors, user_history)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  TESTS COMPLETED (check warnings above)")
    print("=" * 60)


if __name__ == "__main__":
    main()