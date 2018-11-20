import math
import numpy as np

epsilon = 1.e-7


def softmax(x):
    exps = np.exp(x)
    return exps/np.sum(exps, axis=-1)


def kullback_leibler(a, b):
    a = np.clip(a, epsilon, 1.0)
    b = np.clip(b, epsilon, 1.0)
    return np.sum(a * np.log(a / b), axis=-1)


def general_jaccard(a, b):
    _a = np.sum(np.min(np.stack((a, b)), axis=0))
    _b = np.sum(np.max(np.stack((a, b)), axis=0))
    return _a/_b


def softmax_kullback_leibler(a, b):
    a = softmax(a)
    b = softmax(b)
    return np.sum(a * np.log(a / b), axis=-1)


def cross_entropy(a, b):
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, epsilon, 1.0)
    return -np.sum(a * np.log(b), axis=-1)


def softmax_cross_entropy(a, b):
    a = softmax(a)
    b = softmax(b)
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


# %% Testing
if __name__ == "__main__":
    size = 128

    a = np.random.rand(size)
    b = np.random.rand(size)

    print(softmax_logistic_loss(np.array([1., 0., 1.]),
                                np.array([0.99999, 0.00001, 0.99999])))
    print(logistic_loss(np.array([1., 0., 1.]),
                        np.array([0.99999, 0.00001, 0.99999])))
