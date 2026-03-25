import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import os


DATA_DIR = "./data/"
USER_INTERACTION_THRES = 5  # Repos starred by user
REPO_INTERACTION_THRES = 10 # Number of stars
MAX_USERS = 5_000           # Temperory Cap for prototyping
MAX_REPOS = 20_000          # Temperory Cap for prototyping
OUTPUT_DIR = "./filtered_data/"


os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "min_freq_160_preprocessed_data.pkl"), "rb") as f:
    raw_data = pickle.load(f)

# Inspecting the dataset    
# print("Keys:", list(raw_data.keys()) if isinstance(raw_data, dict) else "Not a dict")
# print("Example:", {k: type(v) for k, v in raw_data.items()} if isinstance(raw_data, dict) else "raw_data type")

import sys
sys.path.append(".") # To add spotlight.py to path
from spotlight import Interactions

user_ids = np.asarray(raw_data["user_index2id"])
repo_ids = np.asarray(raw_data["repo_index2id"])

print(user_ids[0])

# TODO Make dataframe
# TODO Make interactions
# TODO save as csv