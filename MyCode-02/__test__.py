import numpy as np

from Database import Utils
from Database import CONFIG
from Schema import SchemaV01
from Estimator import Losses

from tensorflow.keras.callbacks import EarlyStopping

from Generator import AugSiameseV1


shape = CONFIG['mnist']['shape']
X_train, X_test, y_train, y_test = Utils.laod_data('mnist')

X_train = Utils.reshape(X_train / 255.0, shape)
X_test = Utils.reshape(X_test / 255.0, shape)

schema = SchemaV01().buildSiameseV1(shape)

n_cls = CONFIG['mnist']['n_cls']
generator = AugSiameseV1(X_train, y_train, n_cls,
                         datagen_options={
                             'rotation_range': 20,
                             'width_shift_range': 0.2,
                             'height_shift_range': 0.2,
                         })
valid_generator = AugSiameseV1(X_test, y_test, n_cls,
                               datagen_options={
                                   'rotation_range': 20,
                                   'width_shift_range': 0.2,
                                   'height_shift_range': 0.2,
                               })

# for i in generator:
#     i

schema.model.compile(loss=Losses.cross_entropy(),
                     optimizer='adadelta', metrics=['acc'])

history = schema.model.fit_generator(generator, epochs=10,
                                     callbacks=[EarlyStopping(patience=10)],
                                     validation_data=valid_generator)

embeddings = schema.getModel().predict(X_test)

from Estimator.Utils import plot_lda_reduction, plot_pca_reduction, plot_lsa_reduction, plot_lle_reduction

plot_pca_reduction(embeddings, y_test, 'pca')
plot_lsa_reduction(embeddings, y_test, 'lsa')
plot_lda_reduction(embeddings, y_test, 'lda')
# plot_lle_reduction(embeddings, y_test, 'lle')
