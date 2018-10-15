# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pickle
import numpy as np
import itertools as it

from scipy import special

classes_num = 10
indexes_num = [5350, 6405, 4980, 6025, 4790, 5897, 5546, 5731, 5327, 4891]

min_indexes = min(indexes_num)

def _pos_index_generator(rng, r):
    pos_list = np.array(range(rng))
    np.random.shuffle(pos_list)
    return it.combinations(pos_list, r)

def _neg_index_generator(nb_cls, exc, rng, r):
    cls_lst = list(range(nb_cls))
    del cls_lst[cls_lst.index(exc)]
    cls_perm = it.cycle(np.random.permutation(cls_lst))
    nb_pos = special.comb(rng, r, exact=True, repetition=False)
    neg_idx_perm = np.array(range(nb_pos))%rng
    np.random.shuffle(neg_idx_perm)
    pos_idx_perm = np.array(range(nb_pos))%rng
    np.random.shuffle(pos_idx_perm)
    idx_perm = zip(pos_idx_perm, neg_idx_perm)
    return zip(idx_perm, cls_perm)

def _index_generator(nb_cls, exc, rng, r):
    return zip(_pos_index_generator(rng, r),
                    _neg_index_generator(nb_cls, exc, rng, r))

def _class_index_generator(nb_cls, rng, r):
    cls_index = {}
    for i in range(nb_cls):
        cls_index.update({i: []})
        cls_index[i].append(_index_generator(nb_cls, i, rng, r))
    return cls_index

def batch_index_generator(n_split, nb_cls, rng, r):
    nb_pos = special.comb(rng, r, exact=True, repetition=False)
    step = int(nb_pos/n_split)
    cls_index = _class_index_generator(nb_cls, rng, r)
    for cls_id, cls_ind in cls_index.items():
        cls_data = list(cls_ind[0])
        for i in range(n_split):
            with open("./triple_data_v2"+'/'+str(cls_id)+'_'+str(i)+'.pkl', 'wb') as file:    
                pickle.dump(cls_data[i*step:(i+1)*step], file)

batch_index_generator(5, 10, 5000, 2)