# mogp-decomposition

The repository for the UAI 2021 publication 'Multi-output Gaussian Processes for Uncertainty-aware Recommender Systems'.

# Souce code
- [Multi-output / Multi-way GP](./mogp_decomposition/mwgp.py)
- [Data loading functions](./mogp_decomposition/data.py)

# Basic usage
```python
# Define the hyper parameters
hyper_params = {
    'I':I, 'J':J, 'K':K,      # the number of entities in each domain
    'emb_sizes': [8, 8],      # the embedding sizes
    'M': original=128,        # the number of inducing pairs
    'emb_reg': 1e-3,          # l2 norm on the embeddings
    'batch_size': 2**16,      # the size of training batches
    'obs_mean': Y_tr.mean(),  # the mean of target
    'lr': 1e-2                # the learning rate 
}  
gp_md = GPD(**hyper_params)
gp_md.save_path = './ml-1m_lrs/'+str(lr)+'_cv'+str(cv_id)+'/'
gp_md.build()
gp_md.train(X_tr, Y_tr, X_te, Y_te, n_iter=501)
gp_md.save()
```
# Datasets 
- [ML-1M](./data/ML-1M)
- [ML-10M](./data/ML-10M)
- [Jester](./data/Jester)
where each folder also contains the train/test split.

# Experiments. 
- [ML-1M](./experiments/ML-1M)
    - [Training script](./experiments/ML-1M/ML-1M_Training.ipynb)
    - [Results summary](./experiments/ML-1M/ML-1M_Evaluation.ipynb)
- [ML-10M](./experiments/ML-10M)
    - [Training script](./experiments/ML-10M/ml-10m.py)
    - [Results summary](./experiments/ML-10M/ml-10m_summary.ipynb)
- [ML-10M](./experiments/Jester)
    - [Training script](./experiments/Jester/jester.py)
    - [Results summary](./experiments/Jester/jester_summary.ipynb)

