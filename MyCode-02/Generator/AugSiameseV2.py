import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AugSiameseV2(Sequence):
    """
    Which uses the function of contrastive. It is assumed that 0 for the same and 1 for different images.
    """

    def __init__(self, x_set, y_set, n_cls, datagen_options, batch_size=128):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.datagen_options = datagen_options
        self.indices = [np.where(self.y == i)[0] for i in range(n_cls)]
        self.min_len = [self.indices[i].size for i in range(n_cls)]
        self.n_cls = n_cls

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = []
        classes_order = np.random.permutation(range(self.n_cls))
        anchor = classes_order[0]
        other = np.random.choice(classes_order[1:])
        anchor_gen = ImageDataGenerator(**self.datagen_options)
        anchor_gen.fit(self.x)
        other_gen = ImageDataGenerator(**self.datagen_options)
        other_gen.fit(self.x)
        anchor_generator = anchor_gen.flow(*shuffle(self.x[self.indices[anchor]],
                                                    self.y[self.indices[anchor]]),
                                           batch_size=self.batch_size//2)
        other_generator = other_gen.flow(*shuffle(self.x[self.indices[other]],
                                                   self.y[self.indices[other]]),
                                          batch_size=self.batch_size//2)
        for (anchor_x, _), (other_x, _) in zip(anchor_generator, other_generator):
            batch.append((anchor_x, other_x, np.ones((self.batch_size//2))))
            break
        for (anchor_x1, _), (anchor_x2, _) in zip(anchor_generator, anchor_generator):
            batch.append((anchor_x1, anchor_x2, np.zeros((self.batch_size//2))))
            break
        in_1, in_2, out = zip(*batch)
        return [np.concatenate(in_1), np.concatenate(in_2)], np.concatenate(out)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]

