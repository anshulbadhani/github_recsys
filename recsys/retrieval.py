"""
For retrival we will be using FAISS.
So, when we scale to the full dataset we would dnot run into memory / latency issues
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

import faiss

from typing import List, Dict, Tuple
from recsys.config import get_config, Config


class FaissRetriever:
    def __init__(self, config: Config) -> None:
        """
        Parameters:
            - config : Config
                For FaissRetrieval to know about paths of different files for I/O
            - index: faiss.Index*
                The actual FAISS index object
        """

        self.config = config
        self.index = None
        self.repo_ids = None  # to map faiss internal index -> real repo_ids
        self.user_ids = None
        self.repo_embeddings = None  # numpy matrix for FAISS
        self.user_embeddings = None  # numpy matrix
        self.user_embeddings_map = None # Dictionary for fast user lookups
        self.repo_embeddings_map = None
        
        self.is_loaded = False
        
        # Set this to True because we want Cosine Similarity via Inner Product
        self.normalize_embeddings = True 

        if Path(self.config.paths.get_faiss_index_path()).exists():
            self.load_index()
            
        self.__post_init__()
            
    def __post_init__(self):
        self._load_embeddings()
            
    def _load_user_embeddings(self):
        with open(self.config.paths.get_user_embeddings_path(self.config.model.device), "rb") as f:
            user_data = pickle.load(f)
        self.user_embeddings_map = user_data["user_context_vectors"]
        self.user_ids = np.array(list(self.user_embeddings_map.keys()), dtype=np.int64)
        self.user_embeddings = np.array(list(self.user_embeddings_map.values()), dtype="float32")
        
    def _load_repo_embeddgings(self):
        with open(self.config.paths.get_repo_embeddings_path(self.config.model.device), "rb") as f:
            repo_data = pickle.load(f)
            
        # 1. Extract the INNER dictionary
        self.repo_embeddings_map = repo_data["repo_embeddings"]
        
        # 2. Get keys and values from the INNER dictionary
        self.repo_ids = np.array(list(self.repo_embeddings_map.keys()), dtype=np.int64)
        self.repo_embeddings = np.array(list(self.repo_embeddings_map.values()), dtype="float32")
    
    def _load_embeddings(self):
        self._load_repo_embeddgings()
        self._load_user_embeddings()


    def load_index(self):
        """Loads a pre-built FAISS index from disk."""
        print(f"Loading FAISS index from {self.config.paths.get_faiss_index_path()}...")
        self.index = faiss.read_index(str(self.config.paths.get_faiss_index_path()))
        # Load user embeddings into memory for fast lookup during retrieval
        self._load_user_embeddings()

        self.is_loaded = True
        
    def save_index(self):
        """Saves the FAISS index to disk."""
        faiss.write_index(self.index, str(config.paths.get_faiss_index_path()))

    def build_index(self, force_rebuild=False):
        self.is_loaded = False
        if not force_rebuild:
            self.load_index()
            self.is_loaded = True
            return
        
        print("Building FAISS index from scratch...")
        self._load_embeddings()

        if self.normalize_embeddings:
            faiss.normalize_L2(self.repo_embeddings)
        
        base_index = faiss.IndexFlatIP(self.repo_embeddings.shape[1]) # type: ignore
        self.index = faiss.IndexIDMap(base_index)

        # train index if ivf or ivfpq

        self.index.add_with_ids(self.repo_embeddings, self.repo_ids) # type: ignore
        
        self.save_index()
        self.is_loaded = True

    def search_by_user(self, user_id: int, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_loaded:
            raise ValueError("FAISS index is not loaded. Call build_index() first.")

        if user_id not in self.user_embeddings_map:
            print(f"User {user_id} not found in embeddings. Cold start required.")
            return np.array([]), np.array([])

        query_embedding = self.user_embeddings_map[user_id]
        
        # Ensure the vector is float32 for FAISS
        return self.search_by_vector(query_embedding.astype("float32"), k)

    def search_by_vector(self, query_embedding, k=5):
        query = np.array(query_embedding, dtype="float32").reshape(1, -1)

        if not self.normalize_embeddings:
            faiss.normalize_L2(query)

        D, I = self.index.search(query, k) # type: ignore
        # results =[]
        
        # For our ranker
        # for i in range(k):
        #     repo_id = int(I[0][i])
        #     score = float(D[0][i])
        #     if repo_id != -1:  # FAISS returns -1 if it doesn't find enough neighbors
        #         results.append({"repo_id": repo_id, "score": score})
                
        # return results
        return D, I
    
if __name__ == "__main__":
    config = get_config()
    retriever = FaissRetriever(config)
    retriever.build_index(force_rebuild=True)

    # Pick a random repo embedding as query to test vector search
    query = retriever.repo_embeddings[0] # type: ignore

    D, I = retriever.search_by_vector(query, k=100)

    print("Distances:", D)
    print("IDs:", I)
