import numpy as np
from recsys.config import get_config, Config
from recsys.retrieval import FaissRetriever
from recsys.bandits.neuralucb import NeuralUCB
from recsys.bloom_filter import BloomFilter


class GitHubRecommender:
    """
    The main class which actually gives recommendations from the real data.
    This class encapsulates all the classes (Faiss Retrieval, NeuralUCB, BloomFilter etc),
    which are made till now.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.retriever = FaissRetriever(config)
        self.bandit = NeuralUCB(device=config.model.device)
        self.filter = BloomFilter(100000, 0.01)
        self.__post_init__()

    def __post_init__(self):
        """Loads everything into the memeory"""
        self.retriever.build_index(force_rebuild=False)

    def recommend(self, user_id: int, top_k: int = 10):  # add top_k in config.model.top_k
        """
        Generates top-k recommendations for a given user.

        Args:
            user_id: The user to recommend for
            top_k: Number of final recommendations

        Returns:
            Array of top-k repo IDs
        """
        if user_id not in self.retriever.user_ids: # type: ignore
            raise ValueError(f"User {user_id} not found in embeddings")

        user_emb = self.retriever.user_embeddings_map[user_id]  # type: ignore # as it exists becasue of __post_init__() method
        _, candidate_ids = self.retriever.search_by_user(user_id=user_id, k=top_k * 50)

        if candidate_ids.ndim == 2: # type: ignore
            candidate_ids = candidate_ids.flatten()  # (1, k) → (k,) # type: ignore

        if len(candidate_ids) == 0:
            raise RuntimeError("No candidate IDs could be found")

        novel_candidates = self.filter.filter_candidates(user_id=user_id, candidate_ids=candidate_ids)  # type: ignore ;idk why IDE showing type error when the types are literally same!!

        if len(novel_candidates) == 0:
            print(f"Warning: All candidates filtered out for user {user_id}")
            return np.array([])
        
        n_to_score = min(len(novel_candidates), top_k * 15)
        candidates_to_score = novel_candidates[:n_to_score]
        
        repo_embs = []
        valid_candidates = []

        for repo_id in candidates_to_score:
            repo_id_key = int(repo_id)
            if repo_id_key in self.retriever.repo_embeddings_map: # type: ignore
                repo_embs.append(self.retriever.repo_embeddings_map[repo_id_key]) # type: ignore
                valid_candidates.append(repo_id_key)
            else:
                print(f"Warning: Repo {repo_id_key} not found in embeddings")

        if len(repo_embs) == 0:
            raise RuntimeError("No valid repo embeddings found")
        
        repo_embs = np.array(repo_embs)  # Shape: (n_valid, 384)
        valid_candidates = np.array(valid_candidates)  # Shape: (n_valid,)
        

        ucb_scores = self.bandit.score(user_emb=user_emb, item_embs=repo_embs)  # type: ignore
        print(f"\n=== DEBUG User {user_id} ===")
        print(f"Number of candidates scored: {len(ucb_scores)}")
        print(f"UCB Score stats - Min: {ucb_scores.min():.4f}, Max: {ucb_scores.max():.4f}, Std: {ucb_scores.std():.4f}")
        print(f"Top 8 UCB scores : {np.sort(ucb_scores)[-8:][::-1]}")
        print(f"Top 10 recommended repos: {valid_candidates[np.argsort(ucb_scores)[-10:][::-1]]}")

        return self.bandit.pick_top_k(ucb_scores, valid_candidates, k=top_k)

    def record_interaction(self, user_id: int, repo_id: int, reward: float):
        """
        Record a user interaction and update the bandit.
        
        Args:
            user_id: The user
            repo_id: The repo they interacted with
            reward: 1.0 if starred/clicked, 0.0 if ignored
        """
        # Get embeddings
        user_emb = self.retriever.user_embeddings_map[user_id] # type: ignore
        repo_emb = self.retriever.repo_embeddings_map[int(repo_id)] # type: ignore
        
        # Update bandit
        self.bandit.update(user_emb, repo_emb, reward)
        
        # Add to bloom filter if positive interaction
        if reward > 0:
            self.filter.add(user_id, repo_id) # type: ignore


if __name__ == "__main__":
    config = get_config()
    recommender = GitHubRecommender(config)

    # --- SIMULATE A USER SESSION ---

    # Let's grab a real user ID from your dataset
    test_user_id = recommender.retriever.user_ids[4] # type: ignore

    print(f"\n[SESSION START] User {test_user_id} logs in.")

    # 1. Generate initial recommendations
    print("Generating recommendations...")
    recommendations = recommender.recommend(test_user_id, top_k=5)

    print(f"System Recommends: {recommendations}")

    # 2. Simulate User Interaction
    # Let's pretend the user LOVED the first repo (Clicked/Starred),
    # but IGNORED the second repo.
    clicked_repo = recommendations[0]
    ignored_repo = recommendations[1]

    print(f"\n[USER ACTION] User clicks Repo {clicked_repo}!")
    recommender.record_interaction(test_user_id, clicked_repo, reward=1.0)

    print(f"[USER ACTION] User ignores Repo {ignored_repo}.")
    recommender.record_interaction(test_user_id, ignored_repo, reward=0.0)

    # 3. Generate recommendations again (The user refreshes the page)
    print(
        "\n[SESSION REFRESH] User reloads the page. Generating new recommendations..."
    )
    new_recommendations = recommender.recommend(test_user_id, top_k=5)

    print(f"System Recommends: {new_recommendations}")

    # Notice that `clicked_repo` will NOT be in this new list because the Bloom Filter caught it!
    if clicked_repo not in new_recommendations:
        print(
            f"\n✅ Success: Repo {clicked_repo} was filtered out by the Bloom Filter."
        )

    print(
        f"✅ Success: NeuralUCB average loss is now tracking: {recommender.bandit.get_average_loss():.4f}"
    )
