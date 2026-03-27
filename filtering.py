import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import os


# Global Variables
DATA_DIR = "./data/"
USER_INTERACTION_THRES = 5  # Repos starred by user
REPO_INTERACTION_THRES = 10  # Number of stars
MAX_USERS = 5_000  # Temperory Cap for prototyping
MAX_REPOS = 20_000  # Temperory Cap for prototyping
OUTPUT_DIR = "./filtered_data/"
OUTPUT_FILE = "training.csv"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)


os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "min_freq_160_preprocessed_data.pkl"), "rb") as f:
    raw_data = pickle.load(f)

# Inspecting the dataset
# print("Keys:", list(raw_data.keys()) if isinstance(raw_data, dict) else "Not a dict")
# print("Example:", {k: type(v) for k, v in raw_data.items()} if isinstance(raw_data, dict) else "raw_data type")

import sys

sys.path.append(".")  # To add spotlight.py to path
import spotlight
from spotlight import Interactions

repo_index2id = raw_data.get("repo_index2id", None)
user_index2id = raw_data.get("user_index2id", None)
descriptions = raw_data.get("descriptions", {})
lang_list = raw_data.get("id2lang", {}) # id2lang is a list with but lang2id is a dict with indicies as value from lang_list
languages = raw_data.get("languages", None)

print(f"repo_index2id length: {len(repo_index2id) if repo_index2id is not None else 'None'}")
print(f"user_index2id length: {len(user_index2id) if user_index2id is not None else 'None'}")
print(f"Number of descriptions: {len(descriptions)}")


SPLIT_NAME = "train"
interactions = raw_data[SPLIT_NAME]

user_indices = interactions.user_ids
repo_indices = interactions.item_ids

real_user_ids = [user_index2id[idx] for idx in user_indices] if user_indices is not None else user_indices
real_repo_ids = [repo_index2id[idx] for idx in repo_indices] if repo_indices is not None else repo_indices

desc_list = [descriptions.get(rid, "") for rid in real_repo_ids]


# df = pd.DataFrame({
#     "user_id": real_user_ids,
#     "repo_id": real_repo_ids,
#     "language": lang_list,
#     "description": desc_list
# })

# df["description"] = df["description"].astype(str).str.strip()
# df["description"] = df[df["description"].str.len() > 5]

# print("\n=== CSV Summary ===")
# print(f"Split: {SPLIT_NAME}")
# print(f"Interactions: {len(df):,}")
# print(f"Unique users: {df['user_id'].nunique():,}")
# print(f"Unique repos: {df['repo_id'].nunique():,}")

# df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
# print(f"\n✅ Full CSV saved at: {OUTPUT_PATH}")
# print(f"Size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.1f} MB")

# TODO Understand the Interactions and SequenceInteractions object
    # which will be used to test later
# TODO and the AI ass code for dataframe is not working because of unequal lengths of user and repos
    # better to spilt it into two "towers" i.e users and repositories for embeddings generation