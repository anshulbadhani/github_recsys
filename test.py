import pickle
from spotlight import Interactions, SequenceInteractions

with open("data/min_freq_160_preprocessed_data.pkl", "rb") as f:
    raw = pickle.load(f)


# user_ids = raw["train"].user_ids
# item_ids = raw["train"].item_ids
# ratings  = raw["train"].ratings
# interactions = Interactions(user_ids, item_ids, ratings)
interactions = raw["train"] # already of type interactions
interactions.to_sequence(10, 5, 1)
print(interactions.sequences.sequences.shape)
# if hasattr(interactions.sequences, 'targets') and interactions.sequences.targets is not None:
#     print("sequences.targets shape:", interactions.sequences.targets.shape)

print(interactions.test_sequences.sequences[0:2])

# print("Keys:", list(raw.keys()) if isinstance(raw, dict) else "Not a dict")
# print("Example:", {k: type(v) for k, v in raw.items()} if isinstance(raw, dict) else "raw_data type")
# print(raw["train"].__dict__.sequences)