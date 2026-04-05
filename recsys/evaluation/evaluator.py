"""
AI GENERATED
Offline evaluation framework for the recommendation system.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from recsys.config import Config
from recsys.evaluation.metrics import (
    RecommendationMetrics, 
    BanditMetrics, 
    DiversityMetrics
)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    # Ranking metrics
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    map_score: float = 0.0
    
    # Bandit metrics
    cumulative_regret: float = 0.0
    average_reward: float = 0.0
    exploration_rate: float = 0.0
    
    # Diversity metrics
    intra_list_diversity: float = 0.0
    catalog_coverage: float = 0.0
    gini_coefficient: float = 0.0
    
    # Raw data
    per_user_metrics: Dict = field(default_factory=dict)
    all_recommendations: List = field(default_factory=list)
    
    def summary(self) -> str:
        """Return formatted summary."""
        return f"""
╔══════════════════════════════════════════════════════════╗
║              EVALUATION RESULTS SUMMARY                  ║
╠══════════════════════════════════════════════════════════╣
║ RANKING METRICS                                          ║
║  Hit@5:         {self.hit_at_5:6.3f}  │  Hit@10:       {self.hit_at_10:6.3f} ║
║  Precision@5:   {self.precision_at_5:6.3f}  │  Precision@10: {self.precision_at_10:6.3f} ║
║  Recall@5:      {self.recall_at_5:6.3f}  │  Recall@10:    {self.recall_at_10:6.3f} ║
║  NDCG@5:        {self.ndcg_at_5:6.3f}  │  NDCG@10:      {self.ndcg_at_10:6.3f} ║
║  MRR:           {self.mrr:6.3f}  │  MAP:          {self.map_score:6.3f} ║
╠══════════════════════════════════════════════════════════╣
║ BANDIT METRICS                                           ║
║  Cumulative Regret:     {self.cumulative_regret:10.2f}                  ║
║  Average Reward:        {self.average_reward:10.4f}                  ║
║  Exploration Rate:      {self.exploration_rate:10.4f}                  ║
╠══════════════════════════════════════════════════════════╣
║ DIVERSITY METRICS                                        ║
║  Intra-List Diversity:  {self.intra_list_diversity:10.4f}                  ║
║  Catalog Coverage:      {self.catalog_coverage:10.4f}                  ║
║  Gini Coefficient:      {self.gini_coefficient:10.4f}                  ║
╚══════════════════════════════════════════════════════════╝
        """


class OfflineEvaluator:
    """
    Offline evaluation framework using train/test split.
    
    Strategy:
        1. For each user, split their starred repos into train (80%) and test (20%)
        2. Initialize recommender with train set
        3. Generate recommendations
        4. Measure how many test items appear in recommendations
    """
    
    def __init__(
        self, 
        config: Config,
        train_ratio: float = 0.75
    ):
        """
        Args:
            config: Configuration object
            train_ratio: Fraction of user's stars to use for training
        """
        self.config = config
        self.train_ratio = train_ratio
        
        self.rec_metrics = RecommendationMetrics()
        self.bandit_metrics = BanditMetrics()
        self.diversity_metrics = DiversityMetrics()
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load user and repo data."""
        print("Loading evaluation data...")
        
        # ✅ CHANGED: Now uses the TEST users file
        user_path = self.config.paths.data_dir / "processed" / "users" / "min_freq_160_cleaned_users_test.pkl"
        repo_path = self.config.paths.data_dir / "processed" / "repos" / "min_freq_160_cleaned_repos.pkl"
        
        # Load users
        with open(user_path, 'rb') as f:
            self.user_data = pickle.load(f)
        
        # Load repos
        with open(repo_path, 'rb') as f:
            self.repo_data = pickle.load(f)
        
        print(f"Loaded {len(self.user_data)} TEST users and {len(self.repo_data)} repos")
    
    def train_test_split(
        self, 
        user_id: int
    ) -> Tuple[List[int], List[int]]:
        """
        Split user's starred repos into train and test sets.
        
        Args:
            user_id: User to split
            
        Returns:
            (train_repos, test_repos)
        """
        # Support both old and new test file structures
        if isinstance(self.user_data, dict) and "user_history" in self.user_data:
            starred_repos = self.user_data["user_history"].get(user_id, [])
        else:
            starred_repos = self.user_data.get(user_id, {}).get('starred_repos', [])
        
        if len(starred_repos) == 0:
            return [], []
        
        n_repos = len(starred_repos)
        n_train = int(n_repos * self.train_ratio)
        
        # Chronological split (earlier repos for training)
        train_repos = starred_repos[:n_train]
        test_repos = starred_repos[n_train:]
        
        return train_repos, test_repos
    
    def train_bandit(self, recommender, n_users: int = 7000, epochs: int = 30):
        """
        Train NeuralUCB properly using full user history.
        """
        print("\n" + "="*70)
        print("🚀 TRAINING NEURALUCB REWARD ESTIMATOR (Improved)")
        print("="*70)

        # Load full user history
        history_path = str(self.config.paths.get_user_history_path(self.config.data.min_freq))
        with open(history_path, "rb") as f:
            history_data = pickle.load(f)
        
        user_history = history_data.get("user_history", history_data)
        print(f"Training on {len(user_history)} users' full history")

        user_emb_map = recommender.retriever.user_embeddings_map
        repo_emb_map = recommender.retriever.repo_embeddings_map

        user_ids = list(user_history.keys())
        if n_users is not None:
            user_ids = user_ids[:n_users]

        print(f"Selected {len(user_ids)} users for training")

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            batch_user = []
            batch_repo = []
            batch_reward = []

            for user_id in tqdm(user_ids, desc=f"Epoch {epoch+1}"):
                if user_id not in user_emb_map:
                    continue
                user_emb = user_emb_map[user_id]
                starred = user_history[user_id]

                for repo_id in starred:
                    if repo_id in repo_emb_map:
                        batch_user.append(user_emb)
                        batch_repo.append(repo_emb_map[repo_id])
                        batch_reward.append(1.0)

                        if len(batch_reward) >= 4096:   # larger batch
                            recommender.bandit.batch_update(
                                np.array(batch_user, dtype=np.float32),
                                np.array(batch_repo, dtype=np.float32),
                                np.array(batch_reward, dtype=np.float32)
                            )
                            batch_user.clear()
                            batch_repo.clear()
                            batch_reward.clear()

            # Final batch
            if batch_reward:
                recommender.bandit.batch_update(
                    np.array(batch_user, dtype=np.float32),
                    np.array(batch_repo, dtype=np.float32),
                    np.array(batch_reward, dtype=np.float32)
                )

            print(f"  Epoch {epoch+1} finished. Avg loss: {recommender.bandit.get_average_loss():.4f}")

        print(f"\n✅ Training completed! Total updates: {recommender.bandit.update_count:,}")
        print(f"Final average loss: {recommender.bandit.get_average_loss():.4f}\n")
        print(f"Final training stats:")
        print(f"  Total updates : {recommender.bandit.update_count:,}")
        print(f"  Average loss  : {recommender.bandit.get_average_loss():.4f}")
        print(f"  Beta          : {recommender.bandit.beta}")
    
    def evaluate_user(
        self, 
        recommender,
        user_id: int,
        k_values: List[int] = [5, 10]
    ) -> Dict:
        """
        Evaluate recommendations for a single user.
        
        Args:
            recommender: GitHubRecommender instance
            user_id: User to evaluate
            k_values: K values to evaluate at
            
        Returns:
            Dict of metrics
        """
        # Split data
        train_repos, test_repos = self.train_test_split(user_id)
        
        if len(test_repos) == 0:
            return None
        
        # Add train repos to bloom filter (mark as seen)
        for repo_id in train_repos:
            recommender.filter.add(user_id, repo_id)
        
        # Generate recommendations
        try:
            recommendations = recommender.recommend(user_id, top_k=max(k_values))
            print(f"\n=== DEBUG User {user_id} ===")
            print(f"Top 10 recommended: {list(recommendations[:10])}")
            print(f"Ground truth (test): {test_repos[:15]}")
            if hasattr(recommender, 'bandit') and hasattr(recommender.retriever, 'repo_embeddings_map'):
                div = self.diversity_metrics.intra_list_diversity(
                    recommendations[:20], recommender.retriever.repo_embeddings_map
                )
                print(f"Intra-list diversity of this list: {div:.4f}")
        except Exception as e:
            print(f"Error recommending for user {user_id}: {e}")
            return None
        
        if len(recommendations) == 0:
            return None
        
        # Calculate metrics
        test_set = set(test_repos)
        metrics = {}
        
        for k in k_values:
            metrics[f'hit@{k}'] = self.rec_metrics.hit_at_k(recommendations, test_set, k=k)
            metrics[f'precision@{k}'] = self.rec_metrics.precision_at_k(recommendations, test_set, k=k)
            metrics[f'recall@{k}'] = self.rec_metrics.recall_at_k(recommendations, test_set, k=k)
            metrics[f'ndcg@{k}'] = self.rec_metrics.ndcg_at_k(recommendations, test_set, k=k)
        
        metrics['mrr'] = self.rec_metrics.mean_reciprocal_rank(recommendations, test_set)
        metrics['map'] = self.rec_metrics.average_precision(recommendations, test_set)
        
        # Store recommendations for diversity analysis
        metrics['recommendations'] = recommendations
        
        return metrics
    
    def evaluate(
        self, 
        recommender,
        n_users: int = 7000,
        epochs: int = 30,
        k_values: List[int] = [5, 10]
    ) -> EvaluationResults:
        """
        Run full evaluation across multiple users.
        
        Args:
            recommender: GitHubRecommender instance
            n_users: Number of users to evaluate (None = all)
            k_values: K values to evaluate at
            
        Returns:
            EvaluationResults object
        """
        recommender.bandit.load(self.config.paths.get_weights_path())
        # self.train_bandit(recommender, n_users=n_users, epochs=epochs)
        # recommender.bandit.save(self.config.paths.get_weights_path())
        print("\n" + "="*60)
        print("STARTING OFFLINE EVALUATION")
        print("="*60)
        
        # Handle test user structure
        if isinstance(self.user_data, dict) and "user_history" in self.user_data:
            user_ids = list(self.user_data["user_history"].keys())
        else:
            user_ids = list(self.user_data.keys())
        
        if n_users is not None:
            user_ids = user_ids[:n_users]
        
        print(f"Evaluating {len(user_ids)} users...\n")
        
        all_metrics = []
        all_recommendations = []
        recommendation_counts = {}
        
        for user_id in tqdm(user_ids, desc="Evaluating users"):
            user_metrics = self.evaluate_user(recommender, user_id, k_values)
            
            if user_metrics is not None:
                all_metrics.append(user_metrics)
                all_recommendations.append(user_metrics['recommendations'])
                
                # Track recommendation counts for Gini
                for repo_id in user_metrics['recommendations']:
                    recommendation_counts[int(repo_id)] = recommendation_counts.get(int(repo_id), 0) + 1
        
        if len(all_metrics) == 0:
            print("No valid evaluations!")
            return EvaluationResults()
        
        # Aggregate metrics
        results = EvaluationResults()
        
        for k in k_values:
            setattr(results, f'hit_at_{k}', np.mean([m[f'hit@{k}'] for m in all_metrics]))
            setattr(results, f'precision_at_{k}', np.mean([m[f'precision@{k}'] for m in all_metrics]))
            setattr(results, f'recall_at_{k}', np.mean([m[f'recall@{k}'] for m in all_metrics]))
            setattr(results, f'ndcg_at_{k}', np.mean([m[f'ndcg@{k}'] for m in all_metrics]))
        
        results.mrr = np.mean([m['mrr'] for m in all_metrics])
        results.map_score = np.mean([m['map'] for m in all_metrics])
        
        # Diversity metrics
        if len(all_recommendations) > 0 and hasattr(recommender, 'repo_embeddings_map'):
            diversities = []
            for recs in all_recommendations:
                div = self.diversity_metrics.intra_list_diversity(
                    recs, 
                    recommender.retriever.repo_embeddings_map
                )
                diversities.append(div)
            results.intra_list_diversity = np.mean(diversities)
        
        results.catalog_coverage = self.diversity_metrics.catalog_coverage(
            all_recommendations, 
            len(self.repo_data)
        )
        
        results.gini_coefficient = self.diversity_metrics.gini_coefficient(
            recommendation_counts
        )
        
        # Store raw data
        results.per_user_metrics = all_metrics
        results.all_recommendations = all_recommendations
        
        return results


if __name__ == "__main__":
    from recsys.config import get_config
    from recsys.pipeline import GitHubRecommender
    
    # Load config
    config = get_config()
    
    # Initialize recommender
    print("Initializing recommender...")
    recommender = GitHubRecommender(config)
    
    # Run evaluation
    evaluator = OfflineEvaluator(config, train_ratio=0.75)
    results = evaluator.evaluate(recommender, n_users=10)  # Start with just 10 users
    
    # Print results
    print(results.summary())