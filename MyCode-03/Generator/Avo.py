import numpy as np
from itertools import repeat

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical


class Avo(Sequence):

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
            a_index = np.random.randint(self.min_len[anchor])
            p_index = np.random.randint(self.min_len[positive])
            bch_neg = []
            all_neg = classes_order[1:]
            for negetive in all_neg:
                n_index = np.random.randint(self.min_len[negetive])
                bch_neg.append(self.x[self.indices[negetive][n_index]])

            batch.append((self.x[self.indices[anchor][a_index]],
                          self.x[self.indices[positive][p_index]],
                          *bch_neg, np.array([0, *list(repeat(1, self.n_cls-1))]),
                          *to_categorical(classes_order, num_classes=self.n_cls)))
        batch = list(zip(*batch))
        anchor, positive = batch[0], batch[1]
        negetives = batch[2:self.n_cls+1]
        output_dist = batch[self.n_cls+1]

        all_outclass = []
        for item in batch[self.n_cls+2:]:
            all_outclass.append(np.stack(item))

        all_negetive = []
        for item in negetives:
            all_negetive.append(np.stack(item))
        return [np.stack(anchor), np.stack(positive), *all_negetive], [np.stack(output_dist), *all_outclass]

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]
