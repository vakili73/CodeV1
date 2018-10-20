
from .BaseSchema import BaseSchema

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Sequential


class SchemaV01(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV01')
        pass

    def buildConventional(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.input = model.input
        self.output = model.output
        self.model = model
        return self

    def buildSiameseV1(self, shape, distance='l1'):
        """
        The model used in [1]. Which uses the function of cross-entropy. It is assumed that 1 for the same and 0 for different images.

        [1] van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D. 
            (2015). Siamese Neural Networks for One-Shot Image Recognition.
            ICML - Deep Learning Workshop, 7(11), 956–963. 
            https://doi.org/10.1017/CBO9781107415324.004
        """
        model = self.build(shape)
        model.add(layers.Dense(128, activation='sigmoid'))

        self.input = model.input
        self.output = model.output

        input_1 = layers.Input(shape=shape)
        input_2 = layers.Input(shape=shape)

        embedded_1 = model(input_1)
        embedded_2 = model(input_2)

        def output_shape(input_shape):
            return input_shape

        if distance == 'l1':
            distance_layer = layers.Lambda(
                lambda tensors: K.abs(tensors[0] - tensors[1]),
                output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])
        elif distance == 'l2':
            distance_layer = layers.Lambda(
                lambda tensors: K.square(tensors[0] - tensors[1]),
                output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])

        prediction = layers.Dense(1, activation='sigmoid')(distance)

        self.model = Model(inputs=[input_1, input_2], outputs=prediction)
        return self

    def buildSiameseV2(self, shape, distance='l2'):
        """
        The model used in [1]. Which uses the function of cross-entropy. It is assumed that 1 for the same and 0 for different images.

        [1] van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D. 
            (2015). Siamese Neural Networks for One-Shot Image Recognition.
            ICML - Deep Learning Workshop, 7(11), 956–963. 
            https://doi.org/10.1017/CBO9781107415324.004
        """
        model = self.build(shape)
        model.add(layers.Dense(128, activation='sigmoid'))

        self.input = model.input
        self.output = model.output

        input_1 = layers.Input(shape=shape)
        input_2 = layers.Input(shape=shape)

        embedded_1 = model(input_1)
        embedded_2 = model(input_2)

        def output_shape(input_shape):
            return input_shape[0], 1

        if distance == 'l1':
            distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])
        elif distance == 'l2':
            distance_layer = layers.Lambda(
                lambda tensors: K.sqrt(
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True)), output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])

        self.model = Model(inputs=[input_1, input_2], outputs=distance)
        return self

    def buildTriplet(self, shape):
        raise NotImplementedError

    def build(self, shape):
        """
        [1] https://github.com/ajgallego/Clustering-based-k-Nearest-Neighbor/blob/master/utilKerasModels.py

        [2] Gallego, A.-J., Calvo-Zaragoza, J., Valero-Mas, J. J., & Rico-Juan, J. R. 
            (2017). Clustering-based k-Nearest Neighbor Classification for Large-Scale Data with Neural Codes Representation.
            Pattern Recognition, 74, 531–543. 
            https://doi.org/10.1016/j.patcog.2017.09.038
        """
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                                input_shape=shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        return model

    pass
