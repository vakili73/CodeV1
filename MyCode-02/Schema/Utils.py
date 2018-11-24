import os
import numpy as np

from Schema import BaseSchema
from tensorflow.keras.utils import plot_model


# %% Utils function

def load_schema(version: str) -> BaseSchema:
    module = __import__('Schema')
    schema = getattr(module, 'Schema'+version)()
    return schema

def plot_schema(model, title, show_shapes=True,
                show_layer_names=False, rankdir='TB',
                base_path='./logs/schemas'):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    plot_model(model, show_shapes=True,
               to_file=base_path+'/'+title+'.png',
               show_layer_names=show_layer_names,
               rankdir=rankdir)


def save_weights(model, title,
                 base_path='./logs/models'):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = base_path+'/'+title+'_weights.h5'
    model.save_weights(path)


def save_feature(X, y, title,
                 base_path='./logs/features'):
    np.set_printoptions(precision=16)
    if not os.path.exists(base_path):
            os.makedirs(base_path)
    path = base_path+'/'+title+'.txt'
    with open(path, 'w') as f:
        for i in zip(y, X):
            x = ','.join(str(j) for j in i[1])
            f.write(','.join((str(i[0]), x))+'\n')
