"""
For good recommendations, it is necessary to not show the
already seen items multiple times. Therefore, it is necessary
to filter out the items which user has already seen.
"""

import math
import hashlib
import numpy as np
from typing import Tuple


class BloomFilter:
    def __init__(self, expected_items: int, false_positive_rate: float) -> None:
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate

        # m = -n*ln(p) / (ln2)**2; n, p = expected_items, false positive rate
        self.bit_arr_size = int(
            -(self.expected_items * math.log(self.false_positive_rate))
            / (math.log(2) ** 2)
        )

        # k = m*​ln2/n
        self.n_hash_functions = int(
            self.bit_arr_size * math.log(2) / self.expected_items
        )

        self.bit_arr = [False] * self.bit_arr_size

    def _base_hashes(self, item_string: str) -> Tuple[int, int]:
        """
        Kirsch-Mitzenmacher optimization:
        Generates two fundamental hash integer values for a given string.
        Note: We use hashlib instead of Python's built-in hash() because
        Python's hash() changes every time you restart the script!
        """

        item_bytes = item_string.encode("utf-8")

        h1 = int(hashlib.md5(item_bytes).hexdigest(), 16)
        h2 = int(hashlib.sha1(item_bytes).hexdigest(), 16)

        return h1, h2

    def add(self, user_id: np.int64, repo_id: np.int64):
        item_string = f"{user_id}_{repo_id}"
        h1, h2 = self._base_hashes(item_string)

        # for each hash function
        for i in range(self.n_hash_functions):
            index = (h1 + i * h2) % self.bit_arr_size
            self.bit_arr[index] = True

    def check(self, user_id: np.int64, repo_id: np.int64):
        item_string = f"{user_id}_{repo_id}"
        h1, h2 = self._base_hashes(item_string)

        # for each hash, if any index is False.
        # Then it is sure sure that the item has never been seen by the user
        for i in range(self.n_hash_functions):
            index = (h1 + i * h2) % self.bit_arr_size
            if not self.bit_arr[index]:
                return False
        # If all true then the user might have seen it before
        return True

    def filter_candidates(self, user_id: np.int64, candidate_ids: np.ndarray):
        novel_repos = []
        for repo_id in candidate_ids:
            # If the item is NOT in the Bloom filter, it's a safe recommendation
            if not self.check(user_id, repo_id):
                novel_repos.append(repo_id)

        return np.array(novel_repos, dtype=np.int64)


if __name__ == "__main__":
    # Expecting 100 interactions, 1% false positive rate
    bf = BloomFilter(expected_items=100, false_positive_rate=0.01)

    # 1. User 10 stars Repo 404 and 505
    bf.add(10, 404) # type: ignore
    bf.add(10, 505) # type: ignore

    # 2. FAISS retrieves 5 candidates for User 10
    faiss_candidates = np.array([101, 404, 202, 505, 909], dtype=np.int64)

    # 3. Filter them
    print(f"Candidates before filtering: {faiss_candidates}")
    filtered = bf.filter_candidates(user_id=10, candidate_ids=faiss_candidates) # pyright: ignore[reportArgumentType]
    print(f"Candidates after filtering:  {filtered}")
    # Expected output: [101, 202, 909]
