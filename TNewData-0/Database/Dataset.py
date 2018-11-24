import os
import numpy as np
import seaborn as sns
from Database import Info
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

    def summary(self):
        print(self.name)
        print(self.info)
        pass

    def histogram(self, base_path='./logs/datahists'):
        def _hist(title, target):
            unique, counts = np.unique(target, return_counts=True)
            # Set up the matplotlib figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            fig.suptitle(title)
            # Center the data to make it diverging
            y = counts - np.mean(counts)
            sns.barplot(x=unique, y=y, palette="deep", ax=ax1)
            ax1.axhline(0, color="k", clip_on=False)
            ax1.set_ylabel("Diverging")
            # Randomly reorder the data to make it qualitative
            sns.barplot(x=unique, y=counts, palette="deep", ax=ax2)
            ax2.axhline(0, color="k", clip_on=False)
            ax2.set_ylabel("Qualitative")
            # Saving
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
