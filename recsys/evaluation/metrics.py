"""
AI GENERATED
Evaluation metrics for recommendation systems.
"""

import numpy as np
from typing import List, Set, Dict, Tuple
from collections import defaultdict


class RecommendationMetrics:
    """Calculate various recommendation quality metrics."""
    
    @staticmethod
    def hit_at_k(recommended: np.ndarray, relevant: Set[int], k: int = 10) -> float:
        """
        Hit@K: Did we recommend at least 1 relevant item in top-K?
        
        Args:
            recommended: Array of recommended repo IDs (sorted by score)
            relevant: Set of relevant repo IDs (ground truth)
            k: Cutoff position
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        if len(recommended) == 0:
            return 0.0
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & relevant) > 0 else 0.0
    
    @staticmethod
    def precision_at_k(recommended: np.ndarray, relevant: Set[int], k: int = 10) -> float:
        """
        Precision@K: What fraction of top-K are relevant?
        
        Returns:
            Precision in [0, 1]
        """
        if len(recommended) == 0 or k == 0:
            return 0.0
        top_k = set(recommended[:k])
        hits = len(top_k & relevant)
        return hits / k
    
    @staticmethod
    def recall_at_k(recommended: np.ndarray, relevant: Set[int], k: int = 10) -> float:
        """
        Recall@K: What fraction of relevant items did we capture?
        
        Returns:
            Recall in [0, 1]
        """
        if len(recommended) == 0 or len(relevant) == 0:
            return 0.0
        top_k = set(recommended[:k])
        hits = len(top_k & relevant)
        total_relevant = len(relevant)
        return hits / total_relevant
    
    @staticmethod
    def average_precision(recommended: np.ndarray, relevant: Set[int]) -> float:
        """
        Average Precision: Precision averaged at each relevant item position.
        
        Returns:
            AP score in [0, 1]
        """
        if len(relevant) == 0:
            return 0.0
        
        hits = 0
        sum_precisions = 0.0
        
        for i, repo_id in enumerate(recommended):
            if repo_id in relevant:
                hits += 1
                precision_at_i = hits / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / len(relevant)
    
    @staticmethod
    def ndcg_at_k(recommended: np.ndarray, relevant: Set[int], k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K.
        Considers both relevance and position.
        
        Returns:
            NDCG score in [0, 1]
        """
        if len(recommended) == 0:
            return 0.0
        # DCG: sum of (relevance / log2(position + 1))
        dcg = 0.0
        for i, repo_id in enumerate(recommended[:k]):
            if repo_id in relevant:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1
        
        # IDCG: best possible DCG (all relevant items at top)
        idcg = 0.0
        for i in range(min(len(relevant), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(recommended: np.ndarray, relevant: Set[int]) -> float:
        """
        Mean Reciprocal Rank: 1 / (position of first relevant item).
        
        Returns:
            MRR score in [0, 1]
        """
        for i, repo_id in enumerate(recommended):
            if repo_id in relevant:
                return 1.0 / (i + 1)
        return 0.0


class BanditMetrics:
    """Metrics specific to bandit algorithms."""
    
    @staticmethod
    def cumulative_regret(
        observed_rewards: List[float], 
        optimal_rewards: List[float]
    ) -> float:
        """
        Cumulative regret: sum(optimal - observed).
        Lower is better.
        
        Args:
            observed_rewards: Rewards actually received
            optimal_rewards: Rewards from optimal policy (oracle)
            
        Returns:
            Total regret
        """
        return sum(opt - obs for opt, obs in zip(optimal_rewards, observed_rewards))
    
    @staticmethod
    def average_reward(rewards: List[float]) -> float:
        """Average reward over time."""
        return np.mean(rewards) if len(rewards) > 0 else 0.0
    
    @staticmethod
    def exploration_rate(uncertainties: List[float], threshold: float = 0.1) -> float:
        """
        What fraction of recommendations had high uncertainty?
        Measures exploration tendency.
        
        Args:
            uncertainties: Uncertainty values (sigma) for each recommendation
            threshold: Threshold to consider as "exploratory"
            
        Returns:
            Fraction in [0, 1]
        """
        if len(uncertainties) == 0:
            return 0.0
        exploratory = sum(1 for u in uncertainties if u > threshold)
        return exploratory / len(uncertainties)


class DiversityMetrics:
    """Metrics for recommendation diversity."""
    
    @staticmethod
    def intra_list_diversity(
        recommended: np.ndarray, 
        embeddings_map: Dict[int, np.ndarray]
    ) -> float:
        """
        Average pairwise cosine distance within recommendation list.
        Higher = more diverse.
        
        Args:
            recommended: Array of recommended repo IDs
            embeddings_map: Dict mapping repo_id -> embedding
            
        Returns:
            Average diversity in [0, 1]
        """
        if len(recommended) < 2:
            return 0.0
        
        embeddings = []
        for repo_id in recommended:
            repo_id_int = int(repo_id)
            if repo_id_int in embeddings_map:
                embeddings.append(embeddings_map[repo_id_int])
        
        if len(embeddings) < 2:
            return 0.0
        
        embeddings = np.array(embeddings)
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / (norms + 1e-8)
        
        # Compute pairwise cosine similarities
        similarities = embeddings_normalized @ embeddings_normalized.T
        
        # Get upper triangle (avoid diagonal and duplicates)
        n = len(similarities)
        upper_triangle_indices = np.triu_indices(n, k=1)
        pairwise_sims = similarities[upper_triangle_indices]
        
        # Diversity = 1 - similarity
        diversity = 1.0 - np.mean(pairwise_sims)
        
        return diversity
    
    @staticmethod
    def catalog_coverage(
        all_recommendations: List[np.ndarray], 
        total_items: int
    ) -> float:
        """
        What fraction of the catalog was recommended at least once?
        
        Args:
            all_recommendations: List of recommendation arrays from all users
            total_items: Total number of items in catalog
            
        Returns:
            Coverage in [0, 1]
        """
        unique_recommended = set()
        for recs in all_recommendations:
            unique_recommended.update(recs)
        
        return len(unique_recommended) / total_items if total_items > 0 else 0.0
    
    @staticmethod
    def gini_coefficient(recommendation_counts: Dict[int, int]) -> float:
        """
        Gini coefficient: measure of inequality in recommendation distribution.
        0 = perfectly equal, 1 = one item gets all recommendations.
        
        Args:
            recommendation_counts: Dict mapping repo_id -> number of times recommended
            
        Returns:
            Gini coefficient in [0, 1]
        """
        if len(recommendation_counts) == 0:
            return 0.0
        
        counts = np.array(list(recommendation_counts.values()))
        n = len(counts)
        
        # Sort counts
        sorted_counts = np.sort(counts)
        
        # Compute Gini
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        return gini


if __name__ == "__main__":
    # Test metrics
    print("Testing Recommendation Metrics...\n")
    
    recommended = np.array([101, 202, 303, 404, 505, 606, 707, 808, 909, 1010])
    relevant = {202, 303, 999, 1111}  # User actually liked these
    
    metrics = RecommendationMetrics()
    
    print(f"Recommended: {recommended[:5]}...")
    print(f"Relevant (ground truth): {relevant}\n")
    
    print(f"Hit@5:        {metrics.hit_at_k(recommended, relevant, k=5):.3f}")
    print(f"Hit@10:       {metrics.hit_at_k(recommended, relevant, k=10):.3f}")
    print(f"Precision@5:  {metrics.precision_at_k(recommended, relevant, k=5):.3f}")
    print(f"Precision@10: {metrics.precision_at_k(recommended, relevant, k=10):.3f}")
    print(f"Recall@5:     {metrics.recall_at_k(recommended, relevant, k=5):.3f}")
    print(f"Recall@10:    {metrics.recall_at_k(recommended, relevant, k=10):.3f}")
    print(f"NDCG@10:      {metrics.ndcg_at_k(recommended, relevant, k=10):.3f}")
    print(f"MRR:          {metrics.mean_reciprocal_rank(recommended, relevant):.3f}")
    print(f"AP:           {metrics.average_precision(recommended, relevant):.3f}")