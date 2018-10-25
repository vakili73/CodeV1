
from .BaseSchema import BaseSchema

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.backend import epsilon
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, Sequential


class SchemaV03(BaseSchema):
    def __init__(self):
        super().__init__('SchemaV03')
        pass

    def buildConventionalV1(self, shape, n_cls):
        model = self.build(shape)
        layer = layers.Dense(128, activation='relu')
        model.add(layer)
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.extract_layer = 'dense_128_relu'
        self.input = model.input
        self.output = layer.output
        self.model = model
        return self

    def buildConventionalV2(self, shape, n_cls):
        model = self.build(shape)
        layer = layers.Dense(128, activation='sigmoid')
        model.add(layer)
        model.add(layers.Dropout(0.1))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.extract_layer = 'dense_128_sigmoid'
        self.input = model.input
        self.output = layer.output
        self.model = model
        return self

    def buildSiameseV1(self, shape, n_cls, distance='l1'):
        """
        The model used in [1]. Which uses the function of cross-entropy. It is assumed that 1 for the same and 0 for different images.

        [1] van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D. 
            (2015). Siamese Neural Networks for One-Shot Image Recognition.
            ICML - Deep Learning Workshop, 7(11), 956–963. 
            https://doi.org/10.1017/CBO9781107415324.004
        """
        model = self.build(shape)
        model.add(layers.Dense(128, activation='sigmoid'))

        self.extract_layer = 'dense_128_sigmoid'
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

    def buildSiameseV2(self, shape, n_cls, distance='l2'):
        """
        Which uses the function of contrastive. It is assumed that 0 for the same and 1 for different images.

        [1] van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D. 
            (2015). Siamese Neural Networks for One-Shot Image Recognition.
            ICML - Deep Learning Workshop, 7(11), 956–963. 
            https://doi.org/10.1017/CBO9781107415324.004
        """
        model = self.build(shape)
        model.add(layers.Dense(128, activation='sigmoid'))

        self.extract_layer = 'dense_128_sigmoid'
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
                          keepdims=True) + epsilon()), output_shape=output_shape)
            distance = distance_layer([embedded_1, embedded_2])

        self.model = Model(inputs=[input_1, input_2], outputs=distance)
        return self

    def buildTripletV1(self, shape, n_cls, distance='l2'):
        """
        Hoffer, E., & Ailon, N. 
        (2015). Deep metric learning using triplet network. 
        Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9370(2010), 84–92. 
        https://doi.org/10.1007/978-3-319-24261-3_7
        """
        model = self.build(shape)
        model.add(layers.Dense(128, activation='sigmoid'))

        self.extract_layer = 'dense_128_sigmoid'
        self.input = model.input
        self.output = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        embedded_a = model(input_a)
        embedded_p = model(input_p)
        embedded_n = model(input_n)

        def output_shape(input_shape):
            return input_shape[0], 1

        if distance == 'l1':
            pos_distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            neg_distance = neg_distance_layer([embedded_a, embedded_n])
        elif distance == 'l2':
            pos_distance_layer = layers.Lambda(
                lambda tensors: K.sqrt(
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True)), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors: K.sqrt(
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True)), output_shape=output_shape)
            neg_distance = neg_distance_layer([embedded_a, embedded_n])

        concat = layers.Concatenate(axis=-1)([pos_distance, neg_distance])
        softmax = layers.Activation('softmax')(concat)

        self.model = Model(inputs=[input_a, input_p, input_n], outputs=softmax)
        return self

    def buildTripletV2(self, shape, n_cls, distance='l2'):
        """
        Schroff, F., Kalenichenko, D., & Philbin, J. 
        (2015). FaceNet: A unified embedding for face recognition and clustering. 
        In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 07–12–June, pp. 815–823). 
        https://doi.org/10.1109/CVPR.2015.7298682
        """
        model = self.build(shape)
        model.add(layers.Dense(128, activation='sigmoid'))

        self.extract_layer = 'dense_128_sigmoid'
        self.input = model.input
        self.output = model.output

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        input_n = layers.Input(shape=shape)

        embedded_a = model(input_a)
        embedded_p = model(input_p)
        embedded_n = model(input_n)

        def output_shape(input_shape):
            return input_shape[0], 1

        if distance == 'l1':
            pos_distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors: K.sum(K.abs(tensors[0] - tensors[1]), axis=-1,
                                      keepdims=True), output_shape=output_shape)
            neg_distance = neg_distance_layer([embedded_a, embedded_n])
        elif distance == 'l2':
            pos_distance_layer = layers.Lambda(
                lambda tensors:
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors:
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True), output_shape=output_shape)
            neg_distance = neg_distance_layer([embedded_a, embedded_n])

        concat = layers.Concatenate(axis=-1)([pos_distance, neg_distance])

        self.model = Model(inputs=[input_a, input_p, input_n], outputs=concat)
        return self

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
