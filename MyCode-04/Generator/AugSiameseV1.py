import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AugSiameseV1(Sequence):
    """
    Which uses the function of cross-entropy. It is assumed that 1 for the same and 0 for different images.
    """

    def __init__(self, x_set, y_set, n_cls, dgen_opt, batch_size=128):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = [np.where(self.y == i)[0] for i in range(n_cls)]
        self.min_len = [self.indices[i].size for i in range(n_cls)]
        self.n_cls = n_cls
        self.datagen = ImageDataGenerator(**dgen_opt)
        self.datagen.fit(self.x)
        self.generators = []
        for i in range(n_cls):
            generator = self.datagen.flow(self.x[self.indices[i]],
                                          self.y[self.indices[i]],
                                          batch_size=1)
            self.generators.append(generator)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = []
        for _ in range(self.batch_size//2):
            classes_order = np.random.permutation(range(self.n_cls))
            anchor = classes_order[0]
            other = np.random.choice(classes_order[1:])
            anchor_generator = self.generators[anchor]
            other_generator = self.generators[other]
            for (anchor_x, _), (other_x, _) in zip(
                    anchor_generator, other_generator):
                batch.append((anchor_x[0], other_x[0], 0))
                break
            for (anchor_x1, _), (anchor_x2, _) in zip(
                    anchor_generator, anchor_generator):
                batch.append((anchor_x1[0], anchor_x2[0], 1))
                break
        in_1, in_2, out = zip(*batch)
        return [np.stack(in_1), np.stack(in_2)], np.stack(out)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]
        self.generators = []
        for i in range(self.n_cls):
            generator = self.datagen.flow(self.x[self.indices[i]],
                                          self.y[self.indices[i]],
                                          batch_size=1)
            self.generators.append(generator)
