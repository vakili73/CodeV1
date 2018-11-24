import os
import matplotlib.pyplot as plt

# %% History class


class History(object):
    def __init__(self,
                 epoch: list = [],
                 params: dict = {},
                 history: dict = {}):
        self.epoch = epoch
        self.params = params
        self.history = history
        pass

    def summary(self):
        for item in self.params.items():
            print("\t".join(item))

    def plot(self, title, ylim=(0, 3), base_path='./logs/lrcurves'):
        plt.figure()
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.plot(self.epoch,
                 self.history['loss'],
                 label='Train Loss')
        plt.plot(self.epoch,
                 self.history['val_loss'],
                 label='Valid loss')
        plt.ylim(ylim)
        plt.legend()
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+title+'.png'
        plt.gcf().savefig(path)
        return plt.gcf()

    pass


# %% testing
if __name__ == '__main__':
    history = History()
    pass
