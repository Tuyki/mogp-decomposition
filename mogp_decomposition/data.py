"""
Copyright 2021 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Authors:
Yinchong Yang <yinchong.yang@siemens.com>
Florian Buettner <buettner.florian@siemens.com>

"""

import pandas as pd
import os
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


def load_movielens_data_1m(ml_dir): 
    ml_ratings = pd.read_csv(ml_dir + 'ratings.csv', sep=',')
    
    user_IDs = ml_ratings.userId.unique()
    movie_IDs = ml_ratings.movieId.unique()
    time_min = ml_ratings.timestamp.min()
    time_max = ml_ratings.timestamp.max()
    time_bins = np.linspace(time_min, time_max, 45*12)  
    time = np.digitize(ml_ratings.timestamp, time_bins)
    time_IDs = np.unique(time)

    user_dict = dict(zip(user_IDs, np.arange(len(user_IDs))))
    movie_dict = dict(zip(movie_IDs, np.arange(len(movie_IDs))))
    time_dict = dict(zip(time_IDs, np.arange(len(time_IDs))))
    triple_store = np.concatenate([np.array([user_dict[x] for x in ml_ratings.userId.values])[:, None], 
                                   np.array([movie_dict[x] for x in ml_ratings.movieId.values])[:, None], 
                                   np.array([time_dict[x] for x in time])[:, None], 
                                   ml_ratings.rating.values[:, None]], axis=1)
    triple_store = triple_store.astype('int64')

    return triple_store
    

def load_movielens_data_10m(ml_dir): 
    ml_ratings = pd.read_csv(ml_dir + 'ratings.dat', sep='::', header=None, 
                            names=['userId', 'movieId', 'rating', 'timestamp'])
    user_IDs = ml_ratings.userId.unique()
    movie_IDs = ml_ratings.movieId.unique()
    time_min = ml_ratings.timestamp.min()
    time_max = ml_ratings.timestamp.max()
    time_bins = np.linspace(time_min, time_max, 45*12)  
    time = np.digitize(ml_ratings.timestamp, time_bins)
    time_IDs = np.unique(time)

    user_dict = dict(zip(user_IDs, np.arange(len(user_IDs))))
    movie_dict = dict(zip(movie_IDs, np.arange(len(movie_IDs))))
    time_dict = dict(zip(time_IDs, np.arange(len(time_IDs))))
    triple_store = np.concatenate([np.array([user_dict[x] for x in ml_ratings.userId.values])[:, None], 
                                   np.array([movie_dict[x] for x in ml_ratings.movieId.values])[:, None], 
                                   np.array([time_dict[x] for x in time])[:, None], 
                                   ], axis=1)
    triple_store = triple_store.astype('int64')
    
    triple_ratings = ml_ratings.rating.values[:, None]

    return [triple_store, triple_ratings]

 
