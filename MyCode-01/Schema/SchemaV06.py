
from .BaseSchema import BaseSchema

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Sequential


class SchemaV06(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV06')
        pass

    def buildConventional(self, shape, n_cls):
        model = self.build(shape)
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def buildSiamese(self, shape):
        model = self.build(shape)
        layer01 = layers.Dense(128, activation='sigmoid',
                               kernel_regularizer=l2())
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
        [1] Designed by the experimental result and LeNet-5[2] inspiration

        [2] Cun, Y. L., Bottou, L., Bengio, Y., & Haffiner, P. 
            (1998). Gradient based learning applied to document recognition. 
            Proceedings of IEEE, 86(11), 86(11):2278-2324.
        """
        model = Sequential()
        model.add(layers.Conv2D(16, (5, 5), input_shape=shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        return model

    pass
