import os
import numpy as np

from Database import Dataset

from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

# %% Base schema class


class BaseSchema(object):
    def __init__(self,
                 name: str = ''):
        self.name = name
        self.input = NotImplemented
        self.output = NotImplemented
        self.model: Model = Model()
        self.layerEx = {}
        pass

    def _add_layer_ex(self, name, layer):
        self.layerEx.update({name: layer})

    def plot(self, estm, base_path='./logs/schemas'):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        plot_model(self.model,
                   to_file=base_path+'/'+self.name+'_'+estm+'.png',
                   show_shapes=True)

    def build(self, *args, **kwargs):
        raise NotImplementedError

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

    def extract(self, est_name, db: Dataset, path='./logs/features'):
        names = []
        outputs = []
        for name, layer in self.layerEx.items():
            names.append(name)
            outputs.append(layer)
        intermediate_model = Model(self.input, outputs=outputs)
        intermediate_train_feature = intermediate_model.predict(db.X_train)
        self._csv_save(est_name+'_'+db.name+'_'+self.name+'_train', db.y_train, path,
                       zip(names, intermediate_train_feature if len(names) > 1 else [intermediate_train_feature]))
        intermediate_test_feature = intermediate_model.predict(db.X_test)
        self._csv_save(est_name+'_'+db.name+'_'+self.name+'_test', db.y_test, path,
                       zip(names, intermediate_test_feature if len(names) > 1 else [intermediate_test_feature]))

    pass


# %% testing
if __name__ == '__main__':
    baseSchema = BaseSchema()
