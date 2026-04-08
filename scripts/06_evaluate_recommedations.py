"""
AI
Evaluation script - Run offline evaluation on test users.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from recsys.config import get_config
from recsys.pipeline import GitHubRecommender
from recsys.evaluation.evaluator import OfflineEvaluator


def main():
    config = get_config()

    print("🚀 Initializing GitHub Recommender...")
    recommender = GitHubRecommender(config)

    print("📊 Initializing OfflineEvaluator...")
    evaluator = OfflineEvaluator(config, train_ratio=0.75)

    print("\nStarting evaluation on test users...")
    results = evaluator.evaluate(
        recommender,
        n_users=10,          # Balanced for debugging
        epochs=5,
        k_values=[5, 10, 20]
    )

    print(results.summary())

    # Save results
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(output_dir / "evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\n✅ Evaluation completed and saved to {output_dir}/evaluation_results.pkl")


if __name__ == "__main__":
    main()