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
        positive = classes_order[0]
        negetive = np.random.choice(classes_order[1:])
        anchor_gen = ImageDataGenerator(**self.datagen_options)
        anchor_gen.fit(self.x)
        positive_gen = ImageDataGenerator(**self.datagen_options)
        positive_gen.fit(self.x)
        negetive_gen = ImageDataGenerator(**self.datagen_options)
        negetive_gen.fit(self.x)
        anchor_generator = anchor_gen.flow(*shuffle(self.x[self.indices[anchor]],
                                                    self.y[self.indices[anchor]]),
                                           batch_size=self.batch_size)
        positive_generator = positive_gen.flow(*shuffle(self.x[self.indices[positive]],
                                                   self.y[self.indices[positive]]),
                                          batch_size=self.batch_size)
        negetive_generator = negetive_gen.flow(*shuffle(self.x[self.indices[negetive]],
                                                   self.y[self.indices[negetive]]),
                                          batch_size=self.batch_size)
        for (anchor_x, _), (positive_x, _), (negetive_x, _) in zip(anchor_generator, positive_generator, negetive_generator):
            batch.append((anchor_x, positive_x, negetive_x, np.repeat(np.array([0, 1]), self.batch_size)))
            break
        anchor, positive, negetive, output = zip(*batch)
        return [anchor, positive, negetive], output

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]
