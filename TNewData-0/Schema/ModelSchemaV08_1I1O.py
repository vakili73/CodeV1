import os
import numpy as np
import pandas as pd
from Database import Dataset
from Schema import BaseSchema
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, Model, Sequential

# %% 1-INPUT 1-OUTPUT
class ModelSchemaV08_1I1O(BaseSchema):
    def __init__(self):
        super().__init__('ModelSchemaV08_1I1O')
        self.input = NotImplemented
        self.output = NotImplemented
        self.model: Model = Model()
        self.layerEx = {}
        pass

    def plot(self, base_path='./logs/schemas'):
        plot_model(self.model, to_file=base_path+'/'+self.name+'.png')

    def build(self, shape, n_cls):   
        # https://github.com/BIGBALLON/cifar-10-cnn/blob/master/1_Lecun_Network/LeNet_keras.py 
        model = Sequential()
        model.add(layers.Conv2D(6, (5, 5), activation = 'relu', input_shape=shape))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(16, (5, 5), activation = 'relu'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation = 'relu'))
        model.add(layers.Dropout(0.5))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)
        model.add(layers.Dropout(0.1))
        layer02 = layers.Dense(n_cls, activation='softmax')
        model.add(layer02)

        self._add_layer_ex('dense_128_relu', layer01.output)
        self._add_layer_ex('dense_ncls_softmax', layer02.output)

        self.input = model.input
        self.output = model.output
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
