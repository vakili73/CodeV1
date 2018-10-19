import tensorflow as tf

from tensorflow.keras import backend as K


# %% Losses function

def cross_entropy():
    """
    van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D.
    (2015). Siamese Neural Networks for One-Shot Image Recognition.
    ICML - Deep Learning Workshop, 7(11), 956–963. 
    https://doi.org/10.1017/CBO9781107415324.004
    """
    def loss(y_true, y_pred):
        loss = -(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))
        return K.mean(loss)

    return loss


def contrastive(margin=1):
    """
    Hadsell, R., Chopra, S., & LeCun, Y. 
    (2006). Dimensionality reduction by learning an invariant mapping. 
    In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 2, pp. 1735–1742). 
    https://doi.org/10.1109/CVPR.2006.100
    """
    def loss(y_true, y_pred):
        basic_loss = (1 - y_true) * K.square(y_pred) + y_true * \
            K.square(K.maximum(0., margin - y_pred))

        return K.mean(basic_loss)

    return loss


def triplet(alpha=0.2):
    """
    Hoffer, E., & Ailon, N. 
    (2015). Deep metric learning using triplet network. 
    Lecture Notes in Computer Science (Including Subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), 9370(2010), 84–92. 
    https://doi.org/10.1007/978-3-319-24261-3_7
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
