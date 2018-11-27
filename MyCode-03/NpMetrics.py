import math
import numpy as np

from tensorflow.keras import backend as K

epsilon = K.epsilon()


def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps, axis=-1)


def kullback_leibler(a, b):
    a = np.clip(a, epsilon, 1.0)
    b = np.clip(b, epsilon, 1.0)
    return np.sum(a * np.log(a / b), axis=-1)


def general_jaccard_similarity(a, b):
    _a = np.sum(np.min(np.stack((a, b)), axis=0))
    _b = np.sum(np.max(np.stack((a, b)), axis=0))
    return _a/_b


def softmax_kullback_leibler(a, b):
    a = softmax(a)
    b = softmax(b)
    a = np.clip(a, epsilon, 1.0)
    b = np.clip(b, epsilon, 1.0)
    return np.sum(a * np.log(a / b), axis=-1)


def entropy(x):
    x = np.clip(x, epsilon, 1.0)
    return -np.sum(x * np.log(x), axis=-1)


def softmax_entropy(x):
    x = softmax(x)
    x = np.clip(x, epsilon, 1.0)
    return -np.sum(x * np.log(x), axis=-1)


def cross_entropy(a, b):
    a = np.clip(a, epsilon, 1.0)
    b = np.clip(b, epsilon, 1.0)
    return -np.sum(a * np.log(b), axis=-1)


def softmax_cross_entropy(a, b):
    a = softmax(a)
    b = softmax(b)
    b = np.clip(b, epsilon, 1.0)
    return -np.sum(a * np.log(b), axis=-1)


def cross_entropy_loss(a, b):
    return cross_entropy(a, b) +\
        cross_entropy(1.0-a, 1.0-b)


def softmax_cross_entropy_loss(a, b):
    a = softmax(a)
    b = softmax(b)
    return cross_entropy(a, b) +\
        cross_entropy(1.0-a, 1.0-b)


def logistic_loss(a, b):
    return (cross_entropy(a, b) +
            cross_entropy(1.0-a, 1.0-b)) /\
        a.shape[-1]


def softmax_logistic_loss(a, b):
    a = softmax(a)
    b = softmax(b)
    return (cross_entropy(a, b) +
            cross_entropy(1.0-a, 1.0-b)) /\
        a.shape[-1]


def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b), axis=-1))


def cosine_similarity(a, b):
    a = a/np.sqrt(np.sum(np.square(a), axis=-1))
    b = b/np.sqrt(np.sum(np.square(b), axis=-1))
    return np.sum(a * b, axis=-1)


def cosine_distance(a, b):
    a = a/np.sqrt(np.sum(np.square(a), axis=-1))
    b = b/np.sqrt(np.sum(np.square(b), axis=-1))
    return 1.0-np.sum(a * b, axis=-1)


def squared_l2_distance(a, b):
    return np.sum(np.square(a - b), axis=-1)


def softmax_squared_l2_distance(a, b):
    a = softmax(a)
    b = softmax(b)
    return np.sum(np.square(a - b), axis=-1)


def discretize_with_histogram(a, bins):
    _min = np.min(a)
    _max = np.max(a)
    _len_shape = len(a.shape)
    _bins = np.array(range(bins))
    _range = np.linspace(_min, _max, bins+1)
    for _ in range(_len_shape):
        _bins = np.expand_dims(_bins, axis=-1)
        _range = np.expand_dims(_range, axis=-1)
    _cond1 = np.greater_equal(a, _range[:-1])
    _cond2 = np.less(a, _range[1:])
    _cond3 = np.less_equal(a, _range[1:])
    _cond4 = np.concatenate((_cond2[:-1], _cond3[-1:]), axis=0)
    _all_cond = np.all(np.stack((_cond1, _cond4), axis=0), axis=0)
    _axis = tuple([i+1 for i in range(_len_shape)])
    _discrete = np.sum(_all_cond * _bins, axis=0)
    _histogram = np.count_nonzero(_all_cond, axis=_axis)
    return _discrete, _histogram


def joint_histogram(a, bins):
    _uniq_obj = np.zeros((bins, bins, 2, 1))
    for i in range(bins):
        for j in range(bins):
            _uniq_obj[i, j, :, 0] = np.array([i, j])
    _cond = np.all(np.equal(a, _uniq_obj), axis=2)
    return np.count_nonzero(_cond, axis=2)


def mutual_information(a, b, bins=256):
    _a, _ = discretize_with_histogram(a, bins=bins)
    _b, _ = discretize_with_histogram(b, bins=bins)
    ab = np.stack([_a.flatten(), _b.flatten()])
    joint_hist = joint_histogram(ab, bins=bins)
    joint_proba = joint_hist/np.sum(joint_hist)
    joint_proba = np.clip(joint_proba, 1e-7, 1)
    a_proba = np.sum(joint_proba, axis=1)
    b_proba = np.sum(joint_proba, axis=0)
    a_proba = np.expand_dims(a_proba, axis=-1)
    mui = np.sum(joint_hist * joint_proba *
                 np.log(joint_proba / (a_proba*b_proba)))
    return mui


# %% Testing
if __name__ == "__main__":

    a = np.array([
        [0.001, 0.5, 1.25, 0.75, 2.05],
        [0.001, 0.5, 1.25, 0.75, 2.05],
        [0.001, 0.5, 1.25, 0.75, 2.05],
    ])

    discrete, histogram = discretize_with_histogram(a, 2)
    print(discrete, histogram)

    joint_hist = joint_histogram(
        np.stack((discrete.flatten(), discrete.flatten())), 2)
    print(joint_hist)

    print(mutual_information(a.T, a.T, 2))

    # size = 128

    # a = np.random.rand(28, 28, 3)
    # b = np.random.rand(28, 28, 3)

    # image3 = np.array([
    #     [
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1]
    #     ],
    #     [
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1]
    #     ],
    #     [
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1],
    #         [1, 1, 1, 1, 1]
    #     ],
    # ])

    # image4 = np.array([
    #     [
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0]
    #     ],
    #     [
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0]
    #     ],
    #     [
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0]
    #     ],
    # ])

    # print(mutual_information(image3.T, image3.T, 2))
    # print(mutual_information(image3.T, image4.T, 2))
