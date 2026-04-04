"""
NeuralUCB implementation with dropout-based uncertainty estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from recsys.bandits.models import RewardModel


class NeuralUCB:
    """
    Neural Upper Confidence Bound (NeuralUCB) for contextual bandits.

    Uses dropout-based uncertainty estimation for exploration.
    Score = mu(x) + beta * sigma(x)
    where:
        mu(x) = mean predicted reward
        sigma(x) = uncertainty (std of dropout predictions)
        beta = exploration parameter
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: Tuple[int, ...] = (256, 64),
        dropout_rate: float = 0.2,
        beta: float = 0.5,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        n_uncertainty_samples: int = 20,
    ) -> None:
        """
        Args:
            input_dim: Dimension of concatenated [user_emb || item_emb]
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability for uncertainty
            beta: Exploration parameter (higher = more exploration)
            learning_rate: Learning rate for optimizer
            device: 'cpu' or 'cuda'
            n_uncertainty_samples: Number of forward passes for uncertainty estimation
        """

        self.input_dim = input_dim
        self.beta = beta
        self.device = device
        self.n_uncertainty_samples = n_uncertainty_samples

        self.model = RewardModel(
            input_dim=input_dim, hidden_dims=hidden_dims, dropout_rate=dropout_rate
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Training stats
        self.update_count = 0
        self.total_loss = 0.0

    def concatenate_embeddings(
        self, user_emb: np.ndarray, item_embs: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate user embedding with multiple repo embeddings.

        Args:
            user_emb: User embedding, shape (384,)
            item_embs: Repository embeddings, shape (N, 384)

        Returns:
            Concatenated features, shape (N, 768)
        """
        N = len(item_embs)

        # Repeating N times
        user_embs_repeated = np.tile(user_emb, (N, 1))  # (N, 384)
        inputs = np.concatenate([user_embs_repeated, item_embs], axis=1)  # (N, 768)

        return inputs

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict rewards with uncertainty estimation using dropout.

        Args:
            x: Input tensor, shape (N, 768)

        Returns:
            mu: Mean predicted rewards, shape (N,)
            sigma: Uncertainty estimates (std), shape (N,)
        """
        self.model.train()

        predictions = []

        with torch.no_grad():
            for _ in range(self.n_uncertainty_samples):
                pred = self.model(x).squeeze(-1)  # (N,)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)  # (n_samples, N)

        mu = predictions.mean(axis=0)  # (N,)
        sigma = predictions.std(axis=0)  # (N,)

        return mu, sigma

    def compute_ucb_scores(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Compute UCB scores: score = μ + β * σ

        Args:
            mu: Mean predictions, shape (N,)
            sigma: Uncertainties, shape (N,)

        Returns:
            UCB scores, shape (N,)
        """
        return mu + self.beta * sigma

    def score(self, user_emb: np.ndarray, item_embs: np.ndarray) -> np.ndarray:
        """
        Score candidates using NeuralUCB.

        This is the MAIN SCORING FUNCTION.

        Args:
            user_emb: User embedding, shape (384,)
            item_embs: Repository embeddings, shape (N, 384)

        Returns:
            UCB scores, shape (N,)
        """
        inputs = self.concatenate_embeddings(user_emb, item_embs)
        inputs_tensor = torch.FloatTensor(inputs).to(self.device)
        mu, sigma = self.predict_with_uncertainty(inputs_tensor)
        ucb_scores = self.compute_ucb_scores(mu, sigma)

        return ucb_scores

    def pick_top_k(
        self, scores: np.ndarray, candidate_ids: np.ndarray, k: int = 10
    ) -> np.ndarray:
        """
        Select top-k candidates based on scores.

        Args:
            scores: UCB scores, shape (N,)
            candidate_ids: Candidate repo IDs, shape (N,)
            k: Number of recommendations

        Returns:
            Top-k repo IDs, shape (k,)
        """
        top_k_indicies = np.argsort(scores)[::-1][:k]
        return candidate_ids[top_k_indicies]

    def update(self, user_emb: np.ndarray, item_emb: np.ndarray, reward: float):
        """
        Update the model with a single observation.

        Args:
            user_emb: User embedding, shape (384,)
            repo_emb: Repository embedding, shape (384,)
            reward: Observed reward (1.0 if starred, 0.0 otherwise)
        """
        x = np.concatenate([user_emb, item_emb])
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        y_tensor = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(x_tensor)

        loss = self.criterion(pred, y_tensor)
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        self.total_loss += loss.item()

    def batch_update(
        self, user_embs: np.ndarray, repo_embs: np.ndarray, rewards: np.ndarray
    ):
        """
        Update the model with a batch of observations.

        Args:
            user_embs: User embeddings, shape (batch_size, 384)
            repo_embs: Repository embeddings, shape (batch_size, 384)
            rewards: Observed rewards, shape (batch_size,)
        """
        # Concatenate
        x = np.concatenate([user_embs, repo_embs], axis=1)  # (batch_size, 768)
        x_tensor = torch.FloatTensor(x).to(self.device)

        # Target
        y_tensor = (
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        )  # (batch_size, 1)

        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(x_tensor)

        # Compute loss
        loss = self.criterion(pred, y_tensor)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Track stats
        self.total_loss += loss.item() * len(rewards)
        self.update_count += len(rewards)

    def get_average_loss(self) -> float:
        """Get average loss since last reset."""
        if self.update_count == 0:
            return 0.0
        return self.total_loss / self.update_count

    def reset_stats(self):
        """Reset training statistics."""
        self.update_count = 0
        self.total_loss = 0.0

    def save(self, path: Path):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "beta": self.beta,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")

    def load(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint["update_count"]
        self.beta = checkpoint["beta"]
        print(f"Loaded checkpoint from {path}")


if __name__ == "__main__":
    # Test NeuralUCB
    print("Testing NeuralUCB...")

    bandit = NeuralUCB(beta=0.5)

    # Dummy data
    user_emb = np.random.randn(384).astype(np.float32)
    repo_embs = np.random.randn(100, 384).astype(np.float32)
    candidate_ids = np.arange(1000, 1100)

    # 1. Score candidates
    print("\n1. Scoring 100 candidates...")
    scores = bandit.score(user_emb, repo_embs)
    print(f"   Scores shape: {scores.shape}")
    print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # 2. Select top-10
    print("\n2. Selecting top-10...")
    top_10 = bandit.pick_top_k(scores, candidate_ids, k=10)
    print(f"   Top-10 repo IDs: {top_10}")

    # 3. Update with a single observation
    print("\n3. Updating with single observation...")
    bandit.update(user_emb, repo_embs[0], reward=1.0)
    print(f"   Update count: {bandit.update_count}")
    print(f"   Average loss: {bandit.get_average_loss():.4f}")

    # 4. Batch update
    print("\n4. Batch update with 10 observations...")
    batch_user_embs = np.tile(user_emb, (10, 1))
    batch_repo_embs = repo_embs[:10]
    batch_rewards = np.random.randint(0, 2, size=10).astype(np.float32)

    bandit.batch_update(batch_user_embs, batch_repo_embs, batch_rewards)
    print(f"   Update count: {bandit.update_count}")
    print(f"   Average loss: {bandit.get_average_loss():.4f}")

    print("\n✅ All tests passed!")
