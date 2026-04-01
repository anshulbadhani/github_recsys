"""
Step 1: Filter and clean repository metadata.

This script processes raw preprocessed data and extracts clean repository
metadata including names, descriptions, and programming languages.
"""

import pickle
from recsys.config import get_config

config = get_config()

with open(config.paths.get_raw_data_path(config.data.min_freq), "rb") as f:
    raw_data = pickle.load(f)

repo_id2name = raw_data.get("repo_index2id", None)  # the index itself is the id
descriptions = raw_data.get("descriptions", {})  # repo_id = description
lang_map = raw_data.get(
    "id2lang", {}
)  # id2lang is a list with but lang2id is a dict with indicies as value from lang_list
repo_id2lang = raw_data.get("languages", None)
lang_id2name = raw_data.get("id2lang", {})

repos = {}

for repo_id, desc in descriptions.items():
    repo_name = repo_id2name[repo_id]
    repo_lang = lang_id2name[repo_id2lang[repo_id]]
    if repo_lang == None:
        continue
    # print(f"Name: {repo_name}, Language: {repo_lang}")
    prompt = repo_name + " | " + desc + " | " + str(repo_lang)

    repos[repo_id] = {
        "repo_name": repo_name,
        "description": desc,
        "language": repo_lang,
        "prompt": prompt,
    }

print(f"\n✅ Built metadata for {len(repos)} repositories")

# with open(config.paths.get_repo_metadata_path(config.data.min_freq), "wb") as f:
#     pickle.dump({"clean_repos": repos, "num_repos": len(repos)}, f)

print(f"✅ Saved cleaned repo data to: {config.paths.get_repo_metadata_path()}")
