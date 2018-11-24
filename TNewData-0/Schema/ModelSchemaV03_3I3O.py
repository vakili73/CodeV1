import os
import numpy as np
import pandas as pd
from Database import Dataset
from Schema import BaseSchema
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, Model, Sequential

# %% 3-INPUT 3-OUTPUT
class ModelSchemaV03_3I3O(BaseSchema):
    def __init__(self):
        super().__init__('ModelSchemaV03_3I3O')
        self.input = NotImplemented
        self.output = NotImplemented
        self.model: Model = Model()
        self.layerEx = {}
        pass

    def plot(self, base_path='./logs/schemas'):
        plot_model(self.model, to_file=base_path+'/'+self.name+'.png')

    def build(self, shape, n_cls):
        model = Sequential()
        model.add(layers.Dense(4096, activation='relu', input_shape=shape))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)
        processed_a = model(input_a)
        processed_p = model(input_p)
        processed_n = model(input_n)
        concatenate = layers.concatenate([processed_a, processed_p, processed_n])
        model = Model(inputs=[input_a, input_p, input_n], outputs=concatenate)
    
        self.model = model
        pass

    def _add_layer_ex(self, name, layer):
        self.layerEx.update({name: layer})

    def getModel(self) -> Model:
        return self.model

    def getInput(self):
        return self.input

    def getOutput(self):
        return self.output

    def saveWeights(self, file_name,
                     base_path='./logs/models'):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+file_name+'_'+self.name+'_weights.h5'
        self.model.save_weights(path)

    def loadWeights(self, file_path):
        self.model.load_weights(file_path)

    def summary(self):
        self.model.summary()

    def _csv_save(self, name, y_data,
                  base_path, intermediate_feature):
        np.set_printoptions(precision=16)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        for item in intermediate_feature:
            path = base_path+'/'+name+'_'+item[0]+'.txt'
            with open(path, 'a') as f:
                for i in zip(y_data, item[1]):
                    x = ','.join(str(j) for j in i[1])
                    f.write(','.join((str(i[0]), x))+'\n')
        pass

    def extract(self, est_name, db: Dataset, path = './logs/features'):
        names = []
        outputs = []
        for name, layer in self.layerEx.items():
            names.append(name)
            outputs.append(layer)
        intermediate_model = Model(self.input, outputs=outputs)
        intermediate_train_feature = intermediate_model.predict(db.X_train)
        self._csv_save(est_name+'_'+db.name+'_'+self.name+'_train', db.y_train, path,
                       zip(names, intermediate_train_feature))
        intermediate_test_feature = intermediate_model.predict(db.X_test)
        self._csv_save(est_name+'_'+db.name+'_'+self.name+'_test', db.y_test, path,
                       zip(names, intermediate_test_feature))

    pass
