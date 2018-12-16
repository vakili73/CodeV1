import Metrics
import tensorflow as tf

from tensorflow.keras import backend as K


# %% Losses function

def cross_entropy(**kwargs):
    """
    van der Spoel, E., Rozing, M. P., Houwing-Duistermaat, J. J., Eline Slagboom, P., Beekman, M., de Craen, A. J. M., … van Heemst, D.
    (2015). Siamese Neural Networks for One-Shot Image Recognition.
    ICML - Deep Learning Workshop, 7(11), 956–963. 
    https://doi.org/10.1017/CBO9781107415324.004
    """
    def _loss(y_true, y_pred):
        loss = -(y_true*K.log(y_pred)+(1-y_true)*K.log(1-y_pred))
        return loss

    return _loss


def contrastive(margin=1.25, **kwargs):
    """
    Hadsell, R., Chopra, S., & LeCun, Y. 
    (2006). Dimensionality reduction by learning an invariant mapping. 
    In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 2, pp. 1735–1742). 
    https://doi.org/10.1109/CVPR.2006.100
    """
    def _loss(y_true, y_pred):
        loss = (1 - y_true) * K.square(y_pred) + y_true * \
            K.square(K.maximum(0.0, margin - y_pred))
        return loss

    return _loss


def triplet(alpha=0.2, **kwargs):
    """
    Schroff, F., Kalenichenko, D., & Philbin, J. 
    (2015). FaceNet: A unified embedding for face recognition and clustering. 
    In Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 07–12–June, pp. 815–823). 
    https://doi.org/10.1109/CVPR.2015.7298682
    """
    def _loss(y_true, y_pred):
        pos_dist = y_pred[:, 0]
        neg_dist = y_pred[:, 1]
        loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
        return loss

    return _loss


def my_loss(**kwargs):
    n_cls = kwargs['n_cls']
    e_len = kwargs['e_len']

    def _loss(y_true, y_pred):
        embeds_apn = []
        for i in range(len(e_len)):
            _len = i*(e_len[i]*3)
            embed_a = y_pred[:, _len:(_len+e_len[i])]
            embed_p = y_pred[:, (_len+e_len[i]):(_len+(e_len[i]*2))]
            embed_n = y_pred[:, (_len+(e_len[i]*2)):(_len+(e_len[i]*3))]
            embeds_apn.append((embed_a, embed_p, embed_n))

        out_len = 0
        for i in range(len(e_len)):
            out_len += (e_len[i]*3)

        output_a = y_pred[:, out_len:(out_len+n_cls)]
        output_p = y_pred[:, (out_len+n_cls):(out_len+(n_cls*2))]
        output_n = y_pred[:, (out_len+(n_cls*2)):(out_len+(n_cls*3))]

        true_a = y_true[:, :n_cls]
        true_p = y_true[:, n_cls:(n_cls*2)]
        true_n = y_true[:, (n_cls*2):(n_cls*3)]

        zero = K.constant(0, dtype=K.floatx())
        one = K.constant(1, dtype=K.floatx())

        def __loss(anc, pos, neg):
            pos_dist_l2 = Metrics.squared_l2_distance(anc, pos)
            neg_dist_l2 = Metrics.squared_l2_distance(anc, neg)

            pos_dist_kl = Metrics.kullback_leibler(anc, pos) +\
                Metrics.kullback_leibler(pos, anc)
            neg_dist_kl = Metrics.kullback_leibler(anc, neg) +\
                Metrics.kullback_leibler(neg, anc)

            _loss = \
                Metrics.entropy(K.tanh(pos_dist_kl)) +\
                Metrics.entropy(K.tanh(neg_dist_kl)) +\
                Metrics.entropy(K.tanh(pos_dist_l2)) +\
                Metrics.entropy(K.tanh(neg_dist_l2)) +\
                Metrics.cross_entropy(zero, K.tanh(pos_dist_kl)) +\
                Metrics.cross_entropy(one, K.tanh(neg_dist_kl)) +\
                Metrics.cross_entropy(zero, K.tanh(pos_dist_l2)) +\
                Metrics.cross_entropy(one, K.tanh(neg_dist_l2))
            return _loss

        loss = 0
        for i in range(len(e_len)):
            loss += __loss(*embeds_apn[i])
        loss += \
            Metrics.cross_entropy(true_a, output_a) +\
            Metrics.cross_entropy(true_p, output_p) +\
            Metrics.cross_entropy(true_n, output_n)
        return loss

    return _loss
