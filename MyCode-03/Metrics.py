import numpy as np
from tensorflow.keras import backend as K


def kullback_leibler(tensor_a, tensor_b):
    tensor_a = K.clip(tensor_a, K.epsilon(), 1.0)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return K.sum(tensor_a * K.log(tensor_a / tensor_b), axis=-1)


def softmax_kullback_leibler(tensor_a, tensor_b):
    tensor_a = tensor_a/K.sum(tensor_a, axis=-1)
    tensor_b = tensor_b/K.sum(tensor_b, axis=-1)
    tensor_a = K.clip(tensor_a, K.epsilon(), 1.0)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return K.sum(tensor_a * K.log(tensor_a / tensor_b), axis=-1)


def cross_entropy(tensor_a, tensor_b):
    tensor_a = K.clip(tensor_a, 0.0, 1.0)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return -K.sum(tensor_a * K.log(tensor_b), axis=-1)


def softmax_cross_entropy(tensor_a, tensor_b):
    tensor_a = tensor_a/K.sum(tensor_a, axis=-1)
    tensor_b = tensor_b/K.sum(tensor_b, axis=-1)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return -K.sum(tensor_a * K.log(tensor_b), axis=-1)


def cross_entropy_loss(tensor_a, tensor_b):
    return cross_entropy(tensor_a, tensor_b) +\
        cross_entropy(1.0-tensor_a, 1.0-tensor_b)


def softmax_cross_entropy_loss(tensor_a, tensor_b):
    return softmax_cross_entropy(tensor_a, tensor_b) +\
        softmax_cross_entropy(1.0-tensor_a, 1.0-tensor_b)


def logistic_loss(tensor_a, tensor_b):
    return (cross_entropy(tensor_a, tensor_b) +
            cross_entropy(1.0-tensor_a, 1.0-tensor_b)) /\
        tensor_a.shape[-1].value


def softmax_logistic_loss(tensor_a, tensor_b):
    return (softmax_cross_entropy(tensor_a, tensor_b) +
            softmax_cross_entropy(1.0-tensor_a, 1.0-tensor_b)) /\
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
    tensor_a = tensor_a/K.sum(tensor_a, axis=-1)
    tensor_b = tensor_b/K.sum(tensor_b, axis=-1)
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

    print('\ncross_entropy')
    print(K.get_value(cross_entropy(tensor_a, tensor_b)))
    print(K.get_value(cross_entropy(tensor_b, tensor_a)))

    print(K.get_value(cross_entropy(tensor_a, tensor_a)))
    print(K.get_value(cross_entropy(tensor_b, tensor_b)))

    print('\nsoftmax_cross_entropy')
    print(K.get_value(softmax_cross_entropy(tensor_a, tensor_b)))
    print(K.get_value(softmax_cross_entropy(tensor_b, tensor_a)))

    print(K.get_value(softmax_cross_entropy(tensor_a, tensor_a)))
    print(K.get_value(softmax_cross_entropy(tensor_b, tensor_b)))

    print('\ncross_entropy_loss')
    print(K.get_value(cross_entropy_loss(tensor_a, tensor_b)))
    print(K.get_value(cross_entropy_loss(tensor_b, tensor_a)))

    print(K.get_value(cross_entropy_loss(tensor_a, tensor_a)))
    print(K.get_value(cross_entropy_loss(tensor_b, tensor_b)))

    print('\nlogistic_loss')
    print(K.get_value(logistic_loss(tensor_a, tensor_b)))
    print(K.get_value(logistic_loss(tensor_b, tensor_a)))

    print(K.get_value(logistic_loss(tensor_a, tensor_a)))
    print(K.get_value(logistic_loss(tensor_b, tensor_b)))

    print('\ncross_entropy 1-q')
    print(K.get_value(cross_entropy(tensor_a, 1.0-tensor_b)))
    print(K.get_value(cross_entropy(tensor_b, 1.0-tensor_a)))

    print(K.get_value(cross_entropy(tensor_a, 1.0-tensor_a)))
    print(K.get_value(cross_entropy(tensor_b, 1.0-tensor_b)))

    print('\ncosine_similarity')
    print(K.get_value(cosine_similarity(tensor_a, tensor_b)))
    print(K.get_value(cosine_similarity(tensor_b, tensor_a)))

    print(K.get_value(cosine_similarity(tensor_a, tensor_a)))
    print(K.get_value(cosine_similarity(tensor_b, tensor_b)))

    print('\ncosine_distance')
    print(K.get_value(cosine_distance(tensor_a, tensor_b)))
    print(K.get_value(cosine_distance(tensor_b, tensor_a)))

    print(K.get_value(cosine_distance(tensor_a, tensor_a)))
    print(K.get_value(cosine_distance(tensor_b, tensor_b)))

    print('\neuclidean_distance')
    print(K.get_value(euclidean_distance(tensor_a, tensor_b)))
    print(K.get_value(euclidean_distance(tensor_b, tensor_a)))

    print(K.get_value(euclidean_distance(tensor_a, tensor_a)))
    print(K.get_value(euclidean_distance(tensor_b, tensor_b)))

    print('\nsquared_l2_distance')
    print(K.get_value(squared_l2_distance(tensor_a, tensor_b)))
    print(K.get_value(squared_l2_distance(tensor_b, tensor_a)))

    print(K.get_value(squared_l2_distance(tensor_a, tensor_a)))
    print(K.get_value(squared_l2_distance(tensor_b, tensor_b)))
