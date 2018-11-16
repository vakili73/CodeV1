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


def contrastive(margin=1.25):
    """
    Hadsell, R., Chopra, S., & LeCun, Y. 
    (2006). Dimensionality reduction by learning an invariant mapping. 
    In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 2, pp. 1735–1742). 
    https://doi.org/10.1109/CVPR.2006.100
    """
    def loss(y_true, y_pred):
        loss = (1 - y_true) * K.square(y_pred) + y_true * \
            K.square(K.maximum(0.0, margin - y_pred))
        return K.mean(loss)

    return loss


def triplet(alpha=0.4):
    """
    Schroff, F., Kalenichenko, D., & Philbin, J. 
    (2015). FaceNet: A unified embedding for face recognition and clustering. 
    In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 07–12–June, pp. 815–823). 
    https://doi.org/10.1109/CVPR.2015.7298682
    """
    def loss(y_true, y_pred):
        pos_dist = y_pred[:, 0]
        neg_dist = y_pred[:, 1]
        loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
        return K.mean(loss)

    return loss
