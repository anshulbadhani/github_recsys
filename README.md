# TODOs

- [ ] Change the user embeddings to last k or exponential weights
- [ ] Try other indexing like IVF and IVFPQ for full dataset for faster querying
- [ ] Work on Neural Bandit
- [ ] [Optional] Support for dynamic dataset (using github API)
- [ ] It would be better to re-confgiure the config class and have separate sub classes for Retrival, Filtering and Ranking
- [ ] Make a hashtable with separate bloom filters for each user (if planned to scale beyond single user)
- [ ] Make a bloomfilter save and load from memory
- [ ] Make a load_config_from_json for easier config editing
- [ ] Fix all the `# type: ignore` warnings [Optional as it is a VS Code thing ig]


Checkout Design_choices.md for the choices I am considering