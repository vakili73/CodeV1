# pylint: disable=W0614
import tensorflow as tf
from tensorflow.keras import backend as K


def contrastive_loss(margin=1):
    """
    Train a Siamese MLP on pairs of digits from the MNIST dataset.
    It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
    output of the shared network and by optimizing the contrastive loss (see paper
    for mode details).
    [1] "Dimensionality Reduction by Learning an Invariant Mapping"
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Gets to 99.5% test accuracy after 20 epochs.
    3 seconds per epoch on a Titan X GPU
    """
    def loss(y_true, y_pred):
        basic_loss = (1 - y_true) * K.square(y_pred) + y_true * \
            K.square(K.maximum(0., margin - y_pred))

        return K.mean(basic_loss)

    return loss


def triplet_loss(alpha=0.2):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    def loss(y_true, y_pred):
        shape = int(y_pred.shape.as_list()[-1]/3)

        anchor = y_pred[:, 0:shape]
        positive = y_pred[:, shape:2*shape]
        negative = y_pred[:, 2*shape:3*shape]

        # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
        pos_dist = K.sum(K.square(anchor - positive), axis=-1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
        neg_dist = K.sum(K.square(anchor - negative), axis=-1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = pos_dist - neg_dist + alpha
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = K.maximum(basic_loss, 0)

        return K.mean(loss)

    return loss


def arsenault_loss(N, epsilon=1e-7):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    def loss(y_true, y_pred):
        """
        Implementation of the triplet loss function

        Arguments:
        y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
        y_pred -- python list containing three objects:
                anchor -- the encodings for the anchor data
                positive -- the encodings for the positive data (similar to anchor)
                negative -- the encodings for the negative data (different from anchor)
        N  --  The number of dimension 
        beta -- The scaling factor, N is recommended
        epsilon -- The Epsilon value to prevent ln(0)


        Returns:
        loss -- real number, value of the loss
        """
        beta = N
        anchor = tf.convert_to_tensor(y_pred[:, 0:N])
        positive = tf.convert_to_tensor(y_pred[:, N:N*2])
        negative = tf.convert_to_tensor(y_pred[:, N*2:N*3])

        # distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), -1)
        # distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), -1)

        # Non Linear Values

        # -ln(-x/N+1)
        pos_dist = -tf.log(-tf.divide((pos_dist), beta)+1+epsilon)
        neg_dist = -tf.log(-tf.divide((N-neg_dist), beta)+1+epsilon)

        # compute loss
        loss = neg_dist + pos_dist

        return K.mean(loss)

    return loss
