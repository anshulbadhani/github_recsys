# GitHub RecSys: Personalized Repository Recommender

I built this project because I felt that GitHub's native repository recommendations are often not personalized enough to a user's specific evolving interests. This system uses a multi-stage pipeline—**Retrieval, Filtering, and Ranking**—to provide high-quality, novel, and personalized repository suggestions.

Currently, I am working on the **online testing** of this algorithm using my own personal data via the GitHub API.

## Project Structure

The project is divided into a reusable core engine and dataset-specific scripts.

```text
.
├── data/                       # Dataset-specific binary files (embeddings, indices, weights)
│   ├── embeddings/             # Repo and user vector embeddings
│   ├── faiss_index/            # FAISS index for fast retrieval
│   ├── processed/              # Cleaned metadata
│   ├── raw/                    # Raw input files
│   └── weights/                # Trained NeuralUCB model weights
├── recsys/                     # REUSABLE CORE (SRC)
│   ├── bandits/                # Ranking logic using NeuralUCB
│   ├── bloom_filter.py         # Filtering for novelty (avoid seen items)
│   ├── config/                 # Centralized configuration
│   ├── evaluation/             # Metrics and offline evaluation framework
│   ├── pipeline.py             # Main GitHubRecommender orchestrator
│   └── retrieval.py            # FAISS-based candidate selection
├── scripts/                    # ONE-TIME / DATASET-SPECIFIC SCRIPTS
│   ├── 01_filter_repos.py      # Preprocessing metadata
│   ├── 02_filter_users.py      # Cleaning user histories
│   ├── 03_build_repo_embeddings.py # Generating repo vectors
│   ├── 04_build_user_embeddings.py # Generating user vectors
│   └── 06_evaluate_recommedations.py # Running the evaluation suite
├── pyproject.toml              # Dependencies and project metadata
└── uv.lock                     # Lockfile for reproducibility
```

### Core Components (Reusable)

- **`recsys/retrieval.py`**: Uses FAISS (Facebook AI Similarity Search) for efficient candidate retrieval. It maps user embeddings to the top-$K$ most similar repository embeddings in a high-dimensional vector space.
- **`recsys/bloom_filter.py`**: A memory-efficient Bloom Filter ensures that users are never recommended repositories they have already interacted with, maintaining novelty in every session.
- **`recsys/bandits/neuralucb.py`**: Implements **Neural Upper Confidence Bound (NeuralUCB)** for ranking. It uses a neural network with dropout-based uncertainty estimation to balance exploration (trying new things) and exploitation (recommending what the user is likely to like).
- **`recsys/pipeline.py`**: The `GitHubRecommender` class which encapsulates the entire workflow from retrieval to final ranking.

### Scripts (Dataset Specific)

The `scripts/` directory contains tools to process specific datasets (like the one by *Kim et al.* used here). These are generally run once to prepare the `data/` directory for the reusable `recsys` engine.

## Getting Started

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for extremely fast Python package and project management.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anshulbadhani/github_recsys.git
   cd github-recsys
   ```

2. **Sync dependencies:**
   ```bash
   uv sync
   ```

### Configuration

Before running, you may need to update `recsys/config/config.py` to match your dataset paths or model preferences:

```python
# In recsys/config/config.py
@dataclass
class DataConfig:
    min_freq: int = 160  # Adjust based on your dataset filtering
```

### Usage

To generate recommendations for a user in your dataset:

```python
from recsys.config import get_config
from recsys.pipeline import GitHubRecommender

config = get_config()
recommender = GitHubRecommender(config)

# Get top 10 recommendations for a specific user ID
user_id = 12345
recommendations = recommender.recommend(user_id, top_k=10)
print(f"Recommended Repo IDs: {recommendations}")
```

To run the offline evaluation suite:

```bash
uv run scripts/06_evaluate_recommedations.py
```
