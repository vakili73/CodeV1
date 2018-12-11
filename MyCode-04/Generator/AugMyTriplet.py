import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class AugMyTriplet(Sequence):

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
        for _ in range(self.batch_size):
            classes_order = np.random.permutation(range(self.n_cls))
            anchor = classes_order[0]
            positive = classes_order[0]
            negetive = np.random.choice(classes_order[1:])
            anchor_generator = self.generators[anchor]
            positive_generator = self.generators[positive]
            negetive_generator = self.generators[negetive]
            for (anchor_x, _), (positive_x, _), (negetive_x, _) in zip(
                    anchor_generator, positive_generator, negetive_generator):
                batch.append((anchor_x[0], positive_x[0], negetive_x[0],
                              to_categorical(anchor, self.n_cls),
                              to_categorical(positive, self.n_cls),
                              to_categorical(negetive, self.n_cls)))
                break
        in_anc, in_pos, in_neg, out_anc, out_pos, out_neg = zip(*batch)
        return [np.stack(in_anc), np.stack(in_pos), np.stack(in_neg)], np.concatenate([
            np.stack(out_anc), np.stack(out_pos), np.stack(out_neg)], axis=-1)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]
        self.generators = []
        for i in range(self.n_cls):
            generator = self.datagen.flow(self.x[self.indices[i]],
                                          self.y[self.indices[i]],
                                          batch_size=1)
            self.generators.append(generator)
