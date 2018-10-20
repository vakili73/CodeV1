import numpy as np

from Database import Utils
from Database import CONFIG
from Schema import SchemaV01
from Estimator import Losses

from tensorflow.keras.callbacks import EarlyStopping

from Generator import Triplet


shape = CONFIG['mnist']['shape']
X_train, X_test, y_train, y_test = Utils.laod_data('mnist')

X_train = Utils.reshape(X_train / 255.0, shape)
X_test = Utils.reshape(X_test / 255.0, shape)

schema = SchemaV01().buildTripletV1(shape)

n_cls = CONFIG['mnist']['n_cls']
generator = Triplet(X_train, y_train, n_cls)
valid_generator = Triplet(X_test, y_test, n_cls)

schema.model.compile(loss=Losses.triplet(),
                     optimizer='adadelta', metrics=['acc'])

history = schema.model.fit_generator(generator, epochs=10,
                                     callbacks=[EarlyStopping(patience=10)],
                                     validation_data=valid_generator)

embeddings = schema.getModel().predict(X_test)

from Estimator.Utils import plot_lda_reduction, plot_pca_reduction, plot_lsa_reduction, plot_lle_reduction

plot_pca_reduction(embeddings, y_test, 'pca')
plot_lsa_reduction(embeddings, y_test, 'lsa')
plot_lda_reduction(embeddings, y_test, 'lda')
plot_lle_reduction(embeddings, y_test, 'lle')

