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
        self.indices = [np.where(self.y == i)[0] for i in range(n_cls)]
        self.min_len = [self.indices[i].size for i in range(n_cls)]
        self.n_cls = n_cls
        self.datagen = ImageDataGenerator(**datagen_options)
        self.datagen.fit(self.x)
        self.generators = []
        for i in range(n_cls):
            generator = self.datagen.flow(self.x[self.indices[i]],
                                          self.y[self.indices[i]],
                                          batch_size=self.batch_size//2)
            self.generators.append(generator)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def _fix_batch(self, x_batch, gen_batch):
        for tmp_x_batch, _ in gen_batch:
            x_batch = np.concatenate((x_batch, tmp_x_batch))
            if x_batch.shape[0] < self.batch_size//2:
                continue
            elif x_batch.shape[0] > self.batch_size//2:
                x_batch = x_batch[:self.batch_size//2]
                break
            break
        return x_batch

    def __getitem__(self, idx):
        batch = []
        classes_order = np.random.permutation(range(self.n_cls))
        anchor = classes_order[0]
        other = np.random.choice(classes_order[1:])
        anchor_generator = self.generators[anchor]
        other_generator = self.generators[other]
        for (anchor_x, _), (other_x, _) in zip(
                anchor_generator, other_generator):
            if anchor_x.shape[0] < self.batch_size//2:
                anchor_x = self._fix_batch(anchor_x, anchor_generator)
            if other_x.shape[0] < self.batch_size//2:
                other_x = self._fix_batch(other_x, other_generator)
            batch.append((anchor_x, other_x, np.ones((self.batch_size//2))))
            break
        for (anchor_x1, _), (anchor_x2, _) in zip(
                anchor_generator, anchor_generator):
            if anchor_x1.shape[0] < self.batch_size//2:
                anchor_x1 = self._fix_batch(anchor_x1, anchor_generator)
            if anchor_x2.shape[0] < self.batch_size//2:
                anchor_x2 = self._fix_batch(anchor_x2, anchor_generator)
            batch.append((anchor_x1, anchor_x2, np.zeros((self.batch_size//2))))
            break
        in_1, in_2, out = zip(*batch)
        return [np.concatenate(in_1), np.concatenate(in_2)], np.concatenate(out)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]
        self.generators = []
        for i in range(self.n_cls):
            generator = self.datagen.flow(self.x[self.indices[i]],
                                          self.y[self.indices[i]],
                                          batch_size=self.batch_size//2)
            self.generators.append(generator)


