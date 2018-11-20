import numpy as np
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
