import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K


def softmax(tensor):
    exps = K.exp(tensor)
    return exps/K.sum(exps, axis=-1)


def kullback_leibler(tensor_a, tensor_b):
    tensor_a = K.clip(tensor_a, K.epsilon(), 1.0)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return K.sum(tensor_a * K.log(tensor_a / tensor_b), axis=-1)


def general_jaccard_similarity(tensor_a, tensor_b):
    tensor_min = K.sum(K.min(K.stack((tensor_a, tensor_b)), axis=0))
    tensor_max = K.sum(K.max(K.stack((tensor_a, tensor_b)), axis=0))
    return tensor_min/tensor_max


def softmax_kullback_leibler(tensor_a, tensor_b):
    tensor_a = softmax(tensor_a)
    tensor_b = softmax(tensor_b)
    tensor_a = K.clip(tensor_a, K.epsilon(), 1.0)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return K.sum(tensor_a * K.log(tensor_a / tensor_b), axis=-1)


def entropy(tensor):
    tensor = K.clip(tensor, K.epsilon(), 1.0)
    return -K.sum(tensor * K.log(tensor), axis=-1)


def softmax_entropy(tensor):
    tensor = softmax(tensor)
    tensor = K.clip(tensor, K.epsilon(), 1.0)
    return -K.sum(tensor * K.log(tensor), axis=-1)


def cross_entropy(tensor_a, tensor_b):
    tensor_a = K.clip(tensor_a, K.epsilon(), 1.0)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return -K.sum(tensor_a * K.log(tensor_b), axis=-1)


def softmax_cross_entropy(tensor_a, tensor_b):
    tensor_a = softmax(tensor_a)
    tensor_b = softmax(tensor_b)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return -K.sum(tensor_a * K.log(tensor_b), axis=-1)


def cross_entropy_loss(tensor_a, tensor_b):
    return cross_entropy(tensor_a, tensor_b) +\
        cross_entropy(1.0-tensor_a, 1.0-tensor_b)


def softmax_cross_entropy_loss(tensor_a, tensor_b):
    tensor_a = softmax(tensor_a)
    tensor_b = softmax(tensor_b)
    return cross_entropy(tensor_a, tensor_b) +\
        cross_entropy(1.0-tensor_a, 1.0-tensor_b)


def logistic_loss(tensor_a, tensor_b):
    return (cross_entropy(tensor_a, tensor_b) +
            cross_entropy(1.0-tensor_a, 1.0-tensor_b)) /\
        tensor_a.shape[-1].value


def softmax_logistic_loss(tensor_a, tensor_b):
    tensor_a = softmax(tensor_a)
    tensor_b = softmax(tensor_b)
    return (cross_entropy(tensor_a, tensor_b) +
            cross_entropy(1.0-tensor_a, 1.0-tensor_b)) /\
        tensor_a.shape[-1].value


def cosine_similarity(tensor_a, tensor_b):
    tensor_a = K.l2_normalize(tensor_a, axis=-1)
    tensor_b = K.l2_normalize(tensor_b, axis=-1)
    return K.sum(tensor_a * tensor_b, axis=-1)


def cosine_distance(tensor_a, tensor_b):
    tensor_a = K.l2_normalize(tensor_a, axis=-1)
    tensor_b = K.l2_normalize(tensor_b, axis=-1)
    return 1.0-K.sum(tensor_a * tensor_b, axis=-1)


def euclidean_distance(tensor_a, tensor_b):
    return K.sqrt(K.sum(K.square(tensor_a - tensor_b), axis=-1))


def squared_l2_distance(tensor_a, tensor_b):
    return K.sum(K.square(tensor_a - tensor_b), axis=-1)


def softmax_squared_l2_distance(tensor_a, tensor_b):
    tensor_a = softmax(tensor_a)
    tensor_b = softmax(tensor_b)
    return K.sum(K.square(tensor_a - tensor_b), axis=-1)

# failure
def mutual_information(tensor_a, tensor_b, bins=256):
    channel = tensor_a.shape[-1]
    _tensor_ab = K.stack([tensor_a, tensor_b], axis=-1)

    def _replace(tensor):
        new_x = K.zeros(tensor.shape, dtype=tf.uint8)
        _max = K.get_value(K.max(tensor))
        _range = np.round(np.linspace(0, _max, bins+1), decimals=6)
        for i in range(len(_range)-1):
            _cond1 = K.greater_equal(tensor, _range[i]-K.epsilon())
            if i+1 == len(_range)-1:
                _cond2 = K.less_equal(tensor, _range[i+1]+K.epsilon())
            else:
                _cond2 = K.less(tensor, _range[i+1]+K.epsilon())
                i = K.constant(i, dtype=tf.uint8)
            tf.assign(
                new_x[K.all(K.stack((_cond1, _cond2), axis=-1), axis=-1)], i)
        return new_x

    tensor_ab = K.zeros(_tensor_ab.shape, dtype=tf.uint8)
    for c in range(channel):
        tensor_ab[:, :, c, :] = _replace(_tensor_ab[:, :, c, :])

    joint_histogram = K.zeros((bins, bins, channel), dtype=K.floatx())
    for c in range(channel):
        for i in range(bins):
            for j in range(bins):
                bcast = tf.broadcast_to(K.constant([i, j], dtype=tf.uint8),
                                        tensor_ab[:, :, c, :].shape)
                cond = K.equal(tensor_ab[:, :, c, :], bcast)
                joint_histogram[i, j, c] = K.sum(K.all(cond, axis=-1))

    joint_probability = joint_histogram/K.sum(joint_histogram, axis=(0, 1))
    joint_probability = K.clip(joint_probability, K.epsilon(), 1)

    a_marginal_proba = K.sum(joint_probability, axis=1)
    b_marginal_proba = K.sum(joint_probability, axis=0)

    mui = []
    for c in range(channel):
        _mui = []
        for i in range(bins):
            for j in range(bins):
                _mui.append(joint_histogram[i, j, c] * K.sum(
                    joint_probability[i, j, c] *
                    K.log(joint_probability[i, j, c] /
                           (a_marginal_proba[i, c]*b_marginal_proba[j, c]))))
        mui.append(K.sum(_mui))

    return K.mean(mui)


# %% Testing
if __name__ == "__main__":
    size = 128

    a = np.random.rand(size)
    b = np.random.rand(size)

    tensor_a = K.variable(a)
    tensor_b = K.variable(b)

    print('\nkullback_leibler')
    print(K.get_value(kullback_leibler(tensor_a, tensor_b)))
    print(K.get_value(kullback_leibler(tensor_b, tensor_a)))

    print(K.get_value(kullback_leibler(tensor_a, tensor_a)))
    print(K.get_value(kullback_leibler(tensor_b, tensor_b)))
