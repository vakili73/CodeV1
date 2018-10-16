
from .BaseSchema import BaseSchema

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Sequential


class SchemaV09(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV09')
        pass

    def buildConventional(self, shape, n_cls):
        model = self.build(shape)
        layer01 = layers.Dense(128, activation='sigmoid',
                               kernel_regularizer=l2())
        model.add(layer01)
        model.add(layers.Dense(n_cls, activation='sigmoid',
                               kernel_regularizer=l2()))

        self._add_layer_ex('dense_128_sigmoid', layer01.output)

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
        [1] https://github.com/Goldesel23/Siamese-Networks-for-One-Shot-Learning/blob/master/siamese_network.py

        [2] van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D.
            (2015). Siamese Neural Networks for One-Shot Image Recognition.
            ICML - Deep Learning Workshop, 7(11), 956–963.
            https://doi.org/10.1017/CBO9781107415324.004
        """
        model = Sequential()
        model.add(layers.Conv2D(64, (10, 10), activation='relu',
                                kernel_regularizer=l2(), input_shape=shape))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(128, (7, 7), activation='relu',
                                kernel_regularizer=l2()))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(128, (4, 4), activation='relu',
                                kernel_regularizer=l2()))
        model.add(layers.MaxPooling2D())
        model.add(layers.Conv2D(256, (4, 4), activation='relu',
                                kernel_regularizer=l2()))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='sigmoid',
                               kernel_regularizer=l2()))
        return model

    pass
