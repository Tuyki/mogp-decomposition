"""
Copyright 2021 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Authors:
Yinchong Yang <yinchong.yang@siemens.com>
Florian Buettner <buettner.florian@siemens.com>

"""

import sys
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
# import tensorflow_probability as tfp
import gpflow
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import *

sys.path.append('../../')
from models.mwgp import GPD
import os

np.random.seed(11111986)
tf.set_random_seed(11111986)


os.environ["CUDA_VISIBLE_DEVICES"]="2"


js_dat = pd.read_csv(
    '../../data/Jester/jesterfinal151cols.CSV', 
    sep=';', 
    header=None, 
    na_values='99', 
    index_col=0, 
    decimal=',',
    dtype='float32')
js_dat = js_dat.values
nonna_ids = np.where(~np.isnan(js_dat))
nonna_values = js_dat[nonna_ids]
js_triple_store_ids = np.hstack([nonna_ids[0][:, None], nonna_ids[1][:, None]])
js_triple_store_values = nonna_values
    

N = js_triple_store_ids.shape[0]

ids = np.random.choice(np.arange(N), N, replace=False)
splits = np.array_split(ids, 5)

write_out = open('../../data/Jester/splits.pkl', 'wb')
pickle.dump(splits, write_out)
write_out.close()


for cv_id in range(5): 
    te_ids = splits[cv_id]
    tr_ids = []
    for i in range(len(splits)): 
        if i != cv_id: 
            tr_ids.append(splits[i])
    tr_ids = np.concatenate(tr_ids)

    X_tr = js_triple_store_ids[tr_ids][:, 0:2]
    Y_tr = js_triple_store_values[tr_ids]

    X_te = js_triple_store_ids[te_ids][:, 0:2]
    Y_te = js_triple_store_values[te_ids]

    I = js_triple_store_ids[:, 0].max()+1
    J = js_triple_store_ids[:, 1].max()+1
    K = None

    hyper_params = {'I':I, 'J':J, 'K':K,
                    'emb_sizes': [8, 8], 
                    'M': 128, 
                    'emb_reg': 1e-4,
                    'batch_size': 2**16, 
                    'obs_mean': Y_tr.mean(), 
                    'lr': 1e-2}
    gp_md = GPD(**hyper_params)
    gp_md.save_path = './jester_cv'+str(cv_id)+'/'
    gp_md.build()
    gp_md.train(X_tr, Y_tr, X_te, Y_te, n_iter=101)
    gp_md.save()
    
