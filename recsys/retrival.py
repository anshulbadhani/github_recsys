"""
For retrival we will be using FAISS.
So, when we scale to the full dataset we would dnot run into memory / latency issues
"""

from config import get_config, Config
from scipy import differentiate
import faiss
from faiss import Index
from typing import Optional
from pathlib import Path
import pickle
import numpy as np


class FaissRetriver:
    def __init__(self, config: Config) -> None:
        """
        Parameters:
            - config : Config
                For FaissRetriver to know about paths of different files for I/O
            - index: faiss.Index*
                The actual FAISS index object
        """

        self.config = config
        self.index = None
        self.repo_ids = None  # to map faiss internal index -> real repo_ids
        self.repo_embeddings = None  # to load in memory for faster access
        self.user_embeddings = None  # to load in memory for faster access
        self.is_loaded = False
        self.is_normalized = False
        self.is_built = False

        if Path(self.config.paths.faiss_index_dir).exists():
            self.load_index()

    def load_index(self):
        # TODO: Implement load_index funtion
        self.is_loaded = False
        pass

    def build_index(self, force_rebuild=False):
        self.is_loaded = False
        if not force_rebuild:
            self.load_index()
            self.is_loaded = True
            return
        else:
            with open(self.config.paths.get_repo_embeddings_path("cpu"), "rb") as f:
                data = pickle.load(f)

        repo_data = data["repo_embeddings"]  # type: ignore # data is possibly unbound
        self.repo_ids = np.array(list(repo_data.keys()), dtype=np.int64)
        self.repo_embeddings = np.array(list(repo_data.values()), dtype="float32")

        faiss.normalize_L2(self.repo_embeddings)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.repo_embeddings.shape[1]))

        # train index if ivf or ivfpq

        self.index.add_with_ids(self.repo_embeddings, self.repo_ids) # type: ignore
        self.is_loaded = True

    def search(self, query_embedding, k=5):
        query = np.array(query_embedding, dtype="float32").reshape(1, -1)

        if self.is_normalized:
            faiss.normalize_L2(query)

        D, I = self.index.search(query, k) # type: ignore
        return D, I
    
if __name__ == "__main__":
    config = get_config()
    retriever = FaissRetriver(config)
    retriever.build_index(force_rebuild=True)

    # pick a random repo embedding as query
    query = retriever.repo_embeddings[0] # type: ignore

    D, I = retriever.search(query, k=5)

    print("Distances:", D)
    print("IDs:", I)
