import os
import losses
import metrics
import callbacks
import optimizers
import numpy as np
import adaptation as adapt

from tensorflow.keras import Model
from pandas.util.testing import isiterable
from tensorflow.keras.utils import to_categorical

# %% Algorithms informations


algorithms_info = {
    "regular": {
        "loss": 'categorical_crossentropy',
        "metric": ['categorical_accuracy'],
        "callback": ['early_stopping'],
        "optimizer": 'Adadelta',
        "adaptation": 'regular_transform',
        "extract_layer": 'dense_nb_classes_softmax',
    },
    "siamese_double": {
        "loss": 'contrastive_loss',
        "metric": [],
        "callback": ['early_stopping'],
        "optimizer": 'Adadelta',
        "adaptation": 'siamese_double_transform',
        "extract_layer": 'dense_128_relu',
    },
    "siamese_triplet": {
        "loss": 'triplet_loss',
        "metric": [],
        "callback": ['early_stopping'],
        "optimizer": 'Adadelta',
        "adaptation": 'siamese_triplet_transform',
        "extract_layer": 'dense_128_relu',
    },
    "mo_arsenault": {
        "loss": 'arsenault_loss',
        "metric": [],
        "callback": ['early_stopping'],
        "optimizer": 'Adadelta',
        "adaptation": 'regular_transform',
        "extract_layer": 'dense_nb_classes_sigmoid',
    },
}

# %% Algorithm Class


class Algorithm:
    def __init__(self, name, loss, metric, callback, optimizer, adaptation, extract_layer):
        self.name = name
        self.loss = loss
        self.metric = metric
        self.callback = callback
        self.optimizer = optimizer
        self.adaptation = adaptation
        self.extract_layer = extract_layer

    def set_model(self, model):
        self.model = model

    def transform(self, dataset):
        return self.adaptation(dataset)

    def save_weights(self, db, sim_num):
        path = './models'
        if not os.path.exists(path):
            os.makedirs(path)
        path += '/'+self.name+'_'+db+'_'+str(sim_num)+'_weights.h5'
        self.model.save_weights(path)

    def get_and_save_layer(self, transformed_db, dataset=None):
        if self.name == 'regular':
            inter_out = self.model.get_layer(self.extract_layer).output
            intermediate_layer_model = Model(self.model.input, outputs=inter_out)
            x, y = transformed_db.get_train()
            intermediate_output = intermediate_layer_model.predict(x)
            self._csv_save(y, transformed_db.name, intermediate_output, 'train')
            x, y = transformed_db.get_test()
            intermediate_output = intermediate_layer_model.predict(x)
            self._csv_save(y, transformed_db.name, intermediate_output, 'test')
        else:
            inter_out = self.model.layers[-2].get_layer(self.extract_layer).output
            intermediate_layer_model = Model(self.model.layers[-2].input, outputs=inter_out)
            x, y = dataset.get_train()
            intermediate_output = intermediate_layer_model.predict(x)
            self._csv_save(y, dataset.name, intermediate_output, 'train')
            x, y = dataset.get_test()
            intermediate_output = intermediate_layer_model.predict(x)
            self._csv_save(y, dataset.name, intermediate_output, 'test')



    def _csv_save(self, y, db, intermediate_output, data):
        path = './features'
        if not os.path.exists(path):
            os.makedirs(path)
        concat = np.concatenate(
            (y.reshape((y.size, 1)), intermediate_output), axis=1)
        np.savetxt(path+'/' + self.name + '_' + db + '_' + self.extract_layer +
                   '_' + data + '.txt', concat, delimiter=",")

    def evaluate(self, transformed_db):
        x, y = transformed_db.get_train()
        if self.name != 'regular':
            scalar_values = self.model.evaluate(x=x, y=y)
        else:
            scalar_values = self.model.evaluate(x=x, y=to_categorical(y))
        self._print_evaluate('train', scalar_values)
        x, y = transformed_db.get_test()
        if self.name != 'regular':
            scalar_values = self.model.evaluate(x=x, y=y)
        else:
            scalar_values = self.model.evaluate(x=x, y=to_categorical(y))
        self._print_evaluate('test', scalar_values)

    def _print_evaluate(self, data, scalar_values):
        metrics_names = self.model.metrics_names
        if isiterable(scalar_values):
            for item in zip(metrics_names, scalar_values):
                print(data + ': ' + '\t'.join((item[0], str(item[1]))))
        else:
            print(data + ': ' + '\t'.join((metrics_names[0], str(scalar_values))))



# %% Class instantiation function


def get_algorithm(algorithm, dataset):
    assert algorithm in algorithms_info
    if algorithm != 'mo_arsenault':
        loss = getattr(losses, algorithms_info[algorithm]['loss'])
    else:
        loss = getattr(losses, algorithms_info[algorithm]['loss'])(dataset.nb_classes)
    metric = [getattr(metrics, m)
              for m in algorithms_info[algorithm]['metric']]
    callback = [getattr(callbacks, c)()
                for c in algorithms_info[algorithm]['callback']]
    optimizer = getattr(optimizers, algorithms_info[algorithm]['optimizer'])
    adaptation = getattr(adapt, algorithms_info[algorithm]['adaptation'])
    extract_layer = algorithms_info[algorithm]['extract_layer']
    return Algorithm(algorithm, loss, metric, callback, optimizer, adaptation, extract_layer)
