import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence


class SiameseV2(Sequence):
    """
    Which uses the function of contrastive. It is assumed that 0 for the same and 1 for different images.
    """

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
        for _ in range(self.batch_size//2):
            classes_order = np.random.permutation(range(self.n_cls))
            anchor = classes_order[0]
            other = np.random.choice(classes_order[1:])
            o_index = np.random.randint(self.min_len[other])
            a_index = np.random.randint(self.min_len[anchor])
            batch.append((self.x[self.indices[other][o_index]],
                          self.x[self.indices[anchor][a_index]], 1))
            a_index = np.random.randint(self.min_len[anchor], size=2)
            batch.append((self.x[self.indices[anchor][a_index[0]]],
                          self.x[self.indices[anchor][a_index[1]]], 0))
        in_1, in_2, out = zip(*batch)
        return [np.stack(in_1), np.stack(in_2)], np.stack(out)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]

