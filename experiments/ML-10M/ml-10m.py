"""
Copyright 2021 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Authors:
Yinchong Yang <yinchong.yang@siemens.com>
Florian Buettner <buettner.florian@siemens.com>

"""


import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
# import tensorflow_probability as tfp
import gpflow
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import *

sys.path.append('../../mogp_decomposition/')
from mwgp import GPD
from data import load_movielens_data_10m

np.random.seed(11111986)
tf.set_random_seed(11111986)


#cv_id = int(sys.argv[1])
cv_id = 4

os.environ["CUDA_VISIBLE_DEVICES"]="1"


ml_triple_store, ml_triple_ratings = load_movielens_data_10m('../../data/movielens/ML-10M/') 

read_in = open('./splits.pkl', 'rb')
split_ids = pickle.load(read_in)
read_in.close()

# for cv_id in range(5): 
# cv_id = 0

N = ml_triple_store.shape[0]
I = ml_triple_store[:, 0].max()+1
J = ml_triple_store[:, 1].max()+1
K = None

te_ids = split_ids[cv_id]
tr_ids = []
for i in range(len(split_ids)): 
    if i != cv_id: 
        tr_ids.append(split_ids[i])
tr_ids = np.concatenate(tr_ids)

target_scaler = StandardScaler()

X_tr = ml_triple_store[tr_ids][:, 0:2]
# Y_tr = ml_triple_ratings[tr_ids][:, 0] 
Y_tr = target_scaler.fit_transform(ml_triple_ratings[tr_ids][:, 0][:, None]).reshape(-1)

X_te = ml_triple_store[te_ids][:, 0:2]
# Y_te = ml_triple_ratings[te_ids][:, 0] 
Y_te = target_scaler.transform(ml_triple_ratings[te_ids][:, 0][:, None]).reshape(-1)    


from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA

X_tr_coo = coo_matrix((Y_tr.reshape(-1), (X_tr[:, 0], X_tr[:, 1])), shape=(I, J))
X_tr_dense = X_tr_coo.todense()

pca_user = PCA(8)
pca_item = PCA(8)

user_pcs = pca_user.fit_transform(X_tr_dense)
item_pcs = pca_item.fit_transform(X_tr_dense.T)
    
    
hyper_params = {'I':I, 'J':J, 'K':K,
                'emb_sizes': [8, 8], 
                'M': 256, 
                'emb_reg': 1e-1,
                'batch_size': 2**16, 
                'obs_mean': Y_tr.mean(), 
                'lr': 1e-2}

gp_md = GPD(**hyper_params)
gp_md.save_path = './ml-10m_M=128_cv'+str(cv_id)+'/'
gp_md.build()

# pca init -------------------------------------
gp_md.save()
param0 = gp_md.get_weights_params()
param0[0] = user_pcs
param0[1] = item_pcs
with open('./ml-10m_M=128_cv'+str(cv_id)+'/model_params.pkl', 'wb') as f: 
    pickle.dump(param0, f)
gp_md.load_params()
#------------------------------------------------

gp_md.train(X_tr, Y_tr, None, None, n_iter=501)
gp_md.save()


with open('./ml-10m_scaler_cv'+str(cv_id)+'.pkl', 'wb') as f: 
    pickle.dump(target_scaler, f)
