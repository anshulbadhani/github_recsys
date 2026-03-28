import os
import pickle


# Global Variables
DATA_DIR = "./data/"
DATA_FILE = "min_freq_40_preprocessed_data.pkl"
DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

OUTPUT_DIR = "./filtered_data/"
OUTPUT_FILE = "min_freq_40_cleaned_item.pkl"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)


with open(DATA_PATH, "rb") as f:
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
        "prompt": prompt
    }

print(f"\n✅ Built metadata for {len(repos)} repositories")

print("Example (first 3):")

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump({
        "clean_repos": repos,
        "num_repos": len(repos)
    }, f)

print(f"✅ Saved cleaned repo data to: {OUTPUT_PATH}")
