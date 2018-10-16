from .Data import Info

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras import utils

# %% Dataset class


class Dataset(object):
    def __init__(self,
                 name: str = '',
                 X_train: np.ndarray = np.array([]),
                 y_train: np.ndarray = np.array([]),
                 X_test: np.ndarray = np.array([]),
                 y_test: np.ndarray = np.array([])):
        self.name = name
        self.info = Info[name] if name in Info else None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        pass

    def Y_train(self):
        return utils.to_categorical(self.y_train)

    def Y_test(self):
        return utils.to_categorical(self.y_test)

    def get_shape(self):
        return self.info['shape'][1:]

    def get_data(self, way=None, shot=None):
        if way == None and shot == None:
            return self.X_train, self.y_train, self.X_test, self.y_test
        way = self.info['n_cls'] if way == None else way
        shot = 1 if shot == None else shot
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for i in range(way):
            ind = np.where(self.y_train == i)[0]
            X_train.extend(self.X_train[ind[0:shot]])
            y_train.extend(self.y_train[ind[0:shot]])
            X_test.extend(self.X_train[ind[shot:]])
            y_test.extend(self.y_train[ind[shot:]])
            ind = np.where(self.y_test == i)[0]
            X_test.extend(self.X_test[ind])
            y_test.extend(self.y_test[ind])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        return Dataset(self.name, X_train, y_train, X_test, y_test)

    def summary(self):
        print(self.name)
        print(self.info)
        pass

    def histogram(self, base_path='./logs/datahists'):
        def _hist(title, target):
            unique, counts = np.unique(target, return_counts=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            fig.suptitle(title)
            y = counts - np.mean(counts)
            sns.barplot(x=unique, y=y, palette="deep", ax=ax1)
            ax1.axhline(0, color="k", clip_on=False)
            ax1.set_ylabel("Diverging")
            sns.barplot(x=unique, y=counts, palette="deep", ax=ax2)
            ax2.axhline(0, color="k", clip_on=False)
            ax2.set_ylabel("Qualitative")
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            path = base_path+'/'+self.name+'_'+title+'.png'
            fig.savefig(path)
            return fig
        train_hist_fig = _hist('train', self.y_train)
        test_hist_fig = _hist('test', self.y_test)
        return train_hist_fig, test_hist_fig

    pass


# %% testing
if __name__ == '__main__':
    db = Dataset()
