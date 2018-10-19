
from .BaseSchema import BaseSchema

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Sequential


class SchemaV02(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV02')
        pass

    def buildConventional(self, shape, n_cls):
        model = self.build(shape)
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def buildSiamese(self, shape):
        model = self.build(shape)
        layer01 = layers.Dense(128, kernel_regularizer=l2(),
                               activation='sigmoid')
        model.add(layer01)

        self._add_layer_ex('dense_128_sigmoid', layer01.output)

        self.input = model.input
        self.output = model.output

        input_1 = layers.Input(shape=shape)
        input_2 = layers.Input(shape=shape)

        embedded_1 = model(input_1)
        embedded_2 = model(input_2)

        l1_distance_layer = layers.Lambda(
            lambda tensors: K.abs(tensors[0] - tensors[1]))
        l1_distance = l1_distance_layer([embedded_1, embedded_2])

        prediction = layers.Dense(1, kernel_regularizer=l2(),
                                  activation='sigmoid')(l1_distance)

        self.model = Model(inputs=[input_1, input_2], outputs=prediction)
        pass

    def buildTriplet(self, shape):
        return NotImplemented
        pass

    def build(self, shape):
        """
        [1] https://github.com/ajgallego/Clustering-based-k-Nearest-Neighbor/blob/master/utilKerasModels.py

        [2] Gallego, A.-J., Calvo-Zaragoza, J., Valero-Mas, J. J., & Rico-Juan, J. R. 
            (2017). Clustering-based k-Nearest Neighbor Classification for Large-Scale Data with Neural Codes Representation.
            Pattern Recognition, 74, 531–543. 
            https://doi.org/10.1016/j.patcog.2017.09.038
        """
        model = Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=shape))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.2))
        return model

    pass
