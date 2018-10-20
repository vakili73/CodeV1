import numpy as np

from Database import Utils
from Database import CONFIG
from Schema import SchemaV01
from Estimator import Losses

from tensorflow.keras.callbacks import EarlyStopping

from Generator import SiameseV2


shape = CONFIG['mnist']['shape']
X_train, X_test, y_train, y_test = Utils.laod_data('mnist')

X_train = Utils.reshape(X_train / 255.0, shape)
X_test = Utils.reshape(X_test / 255.0, shape)

schema = SchemaV01().buildSiameseV2(shape)

n_cls = CONFIG['mnist']['n_cls']
generator = SiameseV2(X_train, y_train, n_cls)
valid_generator = SiameseV2(X_test, y_test, n_cls)

schema.model.compile(loss=Losses.contrastive(),
                     optimizer='adadelta', metrics=['acc'])

history = schema.model.fit_generator(generator, epochs=1000,
                                     callbacks=[EarlyStopping(patience=10)],
                                     validation_data=valid_generator)
