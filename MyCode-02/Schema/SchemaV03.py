
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
        layer = layers.Dense(128, activation='sigmoid')
        model.add(layer)
        model.add(layers.Dropout(0.5))
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

        [1] Hadsell R, Chopra S, LeCun Y. 
            Dimensionality reduction by learning an invariant mapping. 
            Innull 2006 Jun 17 (pp. 1735-1742). IEEE.
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
                          keepdims=True) + epsilon()), output_shape=output_shape)
            pos_distance = pos_distance_layer([embedded_a, embedded_p])
            neg_distance_layer = layers.Lambda(
                lambda tensors: K.sqrt(
                    K.sum(K.square(tensors[0] - tensors[1]), axis=-1,
                          keepdims=True) + epsilon()), output_shape=output_shape)
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

    def buildMyModelV1(self, shape, n_cls):
        model = self.build(shape)
        model.add(layers.Dense(128, activation='sigmoid'))

        self.extract_layer = 'dense_128_sigmoid'
        self.input = model.input
        self.output = model.output

        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_cls, activation='softmax'))

        self.myModel = model

        input_a = layers.Input(shape=shape)
        input_p = layers.Input(shape=shape)
        inputs_n = []
        for i in range(n_cls-1):
            inputs_n.append(layers.Input(shape=shape))

        output_p = model(input_p)
        outputs_n = []
        for i in range(n_cls-1):
            outputs_n.append(model(inputs_n[i]))

        embed_model = self.getModel()
        embedded_a = embed_model(input_a)
        embedded_p = embed_model(input_p)
        embeddeds_n = []
        for i in range(n_cls-1):
            embeddeds_n.append(embed_model(inputs_n[i]))

        def output_shape(input_shape):
            return input_shape[0], 1

        def cosine_distance(tensor_a, tensor_b):
            l2_norm_a = K.l2_normalize(tensor_a, axis=-1)
            l2_norm_b = K.l2_normalize(tensor_b, axis=-1)
            return 1-K.sum(l2_norm_a * l2_norm_b, axis=-1,
                           keepdims=True)

        distance_layer = layers.Lambda(
            lambda tensors: cosine_distance(tensors[0], tensors[1]),
            output_shape=output_shape)
        pos_distance = distance_layer([embedded_a, embedded_p])

        neg_distances = []
        for item in embeddeds_n:
            neg_distances.append(distance_layer([embedded_a, item]))

        dist_concat = layers.Concatenate(
            axis=-1)([pos_distance, *neg_distances])

        self.model = Model(inputs=[input_a, input_p, *inputs_n],
                           outputs=[dist_concat, output_p, *outputs_n])
        return self

    def build(self, shape):
        """
        [1] Designed by the experimental result and LeNet-5[3] inspiration

        [2] https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py

        [3] Cun, Y. L., Bottou, L., Bengio, Y., & Haffiner, P. 
            (1998). Gradient based learning applied to document recognition. 
            Proceedings of IEEE, 86(11), 86(11):2278-2324.
        """
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), input_shape=shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(32, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        return model

    pass
