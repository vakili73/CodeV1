import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AugTriplet(Sequence):
    """
    Hoffer, E., & Ailon, N. 
    (2015). Deep metric learning using triplet network. 
    Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9370(2010), 84–92. 
    https://doi.org/10.1007/978-3-319-24261-3_7
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
                                          batch_size=self.batch_size)
            self.generators.append(generator)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def _fix_batch(self, x_batch, gen_batch):
        for tmp_x_batch, _ in gen_batch:
            x_batch = np.concatenate((x_batch, tmp_x_batch))
            if x_batch.shape[0] < self.batch_size:
                continue
            elif x_batch.shape[0] > self.batch_size:
                x_batch = x_batch[:self.batch_size]
                break
            break
        return x_batch

    def __getitem__(self, idx):
        classes_order = np.random.permutation(range(self.n_cls))
        anchor = classes_order[0]
        positive = classes_order[0]
        negetive = np.random.choice(classes_order[1:])
        anchor_generator = self.generators[anchor]
        positive_generator = self.generators[positive]
        negetive_generator = self.generators[negetive]
        for (anchor_x, _), (positive_x, _), (negetive_x, _) in zip(
                anchor_generator, positive_generator, negetive_generator):
            if anchor_x.shape[0] < self.batch_size:
                anchor_x = self._fix_batch(anchor_x, anchor_generator)
            if positive_x.shape[0] < self.batch_size:
                positive_x = self._fix_batch(positive_x, positive_generator)
            if negetive_x.shape[0] < self.batch_size:
                negetive_x = self._fix_batch(negetive_x, negetive_generator)
            batch = [anchor_x, positive_x, negetive_x], np.array(
                [[0, 1]]).repeat(self.batch_size, axis=0)
            break
        return batch

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]
        self.generators = []
        for i in range(self.n_cls):
            generator = self.datagen.flow(self.x[self.indices[i]],
                                          self.y[self.indices[i]],
                                          batch_size=self.batch_size)
            self.generators.append(generator)
