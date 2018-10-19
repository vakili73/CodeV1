import numpy as np

from Database import Utils
from Database import CONFIG
from Schema import SchemaV01
from Estimator import Losses

from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping


shape = CONFIG['mnist']['shape']
X_train, X_test, y_train, y_test = Utils.laod_data('mnist')

X_train = Utils.reshape(X_train / 255.0, shape)
X_test = Utils.reshape(X_test / 255.0, shape)

schema = SchemaV01().buildSiameseV1(shape)

from sklearn.utils import shuffle


class SiameseV1(Sequence):

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
            other = np.random.choice(classes_order[1:])
            o_index = np.random.randint(self.min_len[other])
            a_index = np.random.randint(self.min_len[anchor])
            batch.append((self.x[self.indices[other][o_index]],
                          self.x[self.indices[anchor][a_index]], 0))
            a_index = np.random.randint(self.min_len[anchor], size=2)
            batch.append((self.x[self.indices[anchor][a_index[0]]],
                          self.x[self.indices[anchor][a_index[1]]], 1))
        in_1, in_2, out = zip(*batch)
        return [np.stack(in_1), np.stack(in_2)], np.stack(out)

    def on_epoch_end(self):
        self.x, self.y = shuffle(self.x, self.y)
        self.indices = [np.where(self.y == i)[0] for i in range(self.n_cls)]


n_cls = CONFIG['mnist']['n_cls']
generator = SiameseV1(X_train, y_train, n_cls)
valid_generator = SiameseV1(X_test, y_test, n_cls)

schema.model.compile(loss=Losses.cross_entropy(),
                     optimizer='adadelta', metrics=['acc'])

history = schema.model.fit_generator(generator, epochs=1000,
                                     callbacks=[EarlyStopping(patience=20)],
                                     validation_data=valid_generator)
