
from .BaseSchema import BaseSchema

from tensorflow.keras import layers, Sequential


class SchemaV07(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV07')
        pass

    def buildConvenient(self, shape, n_cls):
        model = self.build(shape)
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

    def buildSiamese(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dropout(0.5))
        layer01 = layers.Dense(128, activation='relu')
        model.add(layer01)

        self._add_layer_ex('dense_128_relu', layer01.output)

        self.input = model.input
        self.output = model.output
        self.model = model
        pass

    def build(self, shape):   
        # https://github.com/BIGBALLON/cifar-10-cnn/blob/master/1_Lecun_Network/LeNet_keras.py 
        model = Sequential()
        model.add(layers.Conv2D(6, (5, 5), activation = 'relu', input_shape=shape))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(16, (5, 5), activation = 'relu'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation = 'relu'))
        return model

    pass
