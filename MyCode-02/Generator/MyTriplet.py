import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


class MyTriplet(Sequence):

    def __init__(self, x_set, y_set, n_cls, batch_size=128):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = [np.where(self.y == i)[0] for i in range(n_cls)]
        self.min_len = [self.indices[i].size for i in range(n_cls)]
        self.n_cls = n_cls

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = []
        for _ in range(self.batch_size):
            classes_order = np.random.permutation(range(self.n_cls))
            anchor = classes_order[0]
            positive = classes_order[0]
            negetive = np.random.choice(classes_order[1:])
            a_index = np.random.randint(self.min_len[anchor])
            p_index = np.random.randint(self.min_len[positive])
            n_index = np.random.randint(self.min_len[negetive])
            batch.append((self.x[self.indices[anchor][a_index]],
                          self.x[self.indices[positive][p_index]],
                          self.x[self.indices[negetive][n_index]],
                          np.array([0., 1.]), to_categorical(
                              classes_order[0], num_classes=self.n_cls)))
        batch = list(zip(*batch))
        anchor = batch[0]
        positive = batch[1]
        negetive = batch[2]
        out_dist = np.stack(batch[3])
        out_pos = np.stack(batch[4])
        return [np.stack(anchor), np.stack(positive),
                np.stack(negetive)], [out_pos, out_dist, out_dist]

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]
