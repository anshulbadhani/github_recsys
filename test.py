import pickle
from spotlight import Interactions, SequenceInteractions

with open("data/min_freq_160_preprocessed_data.pkl", "rb") as f:
    raw = pickle.load(f)

print("Keys:", list(raw.keys()) if isinstance(raw, dict) else "Not a dict")
# print("Example:", {k: type(v) for k, v in raw.items()} if isinstance(raw, dict) else "raw_data type")
print(raw["train"].__dict__["test_sequences"].sequences.ndim)