# Design Choices
I am writing this file to help me keep things structured

## The idea
The idea is to use Two-Tower for Retrival and NeuralUCB for Ranking.

## The pipeline
### 1. Request and Context Formulation
Suppose a request comes from user `anshulbadhani`, the first task this system has to do is gather the user information, like user repositories, preffered programming languages, interests etc. and build a context vector for our Ranker.

For this purpose, we could either use:
- "hardcoded" features (which we will be using for prototyping), which in best case would update everyday offline (if deployed)
- or hardcoded features with current session's context or how `pinterest` does it with long and short term contexts

### 2. Retrival
Our database (about 8k for prototype and about 100k for full training data), we could not rank everything for even 10 users, specially on hardware which I have access to. So, we need to filter the obvious ones out. We will use FAISS by Meta
- For prototyping I will be using `FlatIndexIP` which is for similarity search which compares each item by each item. Which is okay for small dataset like prototype ones, but not for larger ones
- Later we can switch to IVF or IVFPQ for ANN search

### 2.5. Filtering of Ranked items
Items which are already stared by the user or belongs to the user themselves or has interacted before must not be shown again and can be removed from ranked list

### 3. Ranking
Now, we have to rank based on the past user interactions. For that we will be using `LinUCB` for quick prototyping and `NeuralUCB` for actual one, although depending upon my analysis I might look into `LinTS` or `NeuralTS` or simple epsilon-greedy (if needed).


### 4. Serving and Online Feedback
Here comes the system engg part, like how to deploy all of this nd all. Then comes updating our Rankers to get better recomendations. Which I am planning to run offline every few hours. Else it would be too much engg for a quick project

## External Libraries
*Numpy, Torch the standard ones I am not including, only the ones which are new to me are included (well Nothing more than np is req tbh)*
- FAISS
- PySpark (it is like SQL but with python... good stuff, if I scale this too much)

**And I am thinking to not include any bandit libraries as such. As I think it is trivial enough to do it if I can and would help me in my semester project too :skull:**


## Dumb Issues I am encountering
- How and when to write into the faiss.index file? Such that I dont store duplicate repositories in my db