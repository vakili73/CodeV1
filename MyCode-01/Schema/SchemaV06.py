
from .BaseSchema import BaseSchema

from tensorflow.keras import layers, Sequential


class SchemaV06(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV06')
        pass

    def buildConventional(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dropout(0.3))
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

    def buildSiamese(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dropout(0.3))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def buildTriplet(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dropout(0.3))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def build(self, shape):
        # https://github.com/atrybyme/cifar-10-cnn/blob/master/le_net%20.py
                # build the model,(original le_net did not have dropout but we added it to avoid overfitting)
        model = Sequential()
        model.add(layers.Conv2D(16, (5, 5), input_shape=shape,
                                activation='relu', padding='valid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='valid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(32, (4, 4), activation='relu', padding='valid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        return model

    pass
