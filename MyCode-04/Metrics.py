import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K


def top_k_accuracy(y_score, y_true, k=5):
    argsrt = np.argsort(y_score)[:, -k:]
    top_k_bool = []
    for i in range(len(y_true)):
        if y_true[i] in argsrt[i]:
            top_k_bool.append(1)
        else:
            top_k_bool.append(0)
    return np.mean(top_k_bool)


def my_accu_v1(y_true, y_pred, n_cls, e_len=128):
    out_len = (e_len*3)
    output_a = y_pred[:, out_len:(out_len+n_cls)]
    output_p = y_pred[:, (out_len+n_cls):(out_len+(n_cls*2))]
    output_n = y_pred[:, (out_len+(n_cls*2)):(out_len+(n_cls*3))]
    true_a = y_true[:, :n_cls]
    true_p = y_true[:, n_cls:(n_cls*2)]
    true_n = y_true[:, (n_cls*2):(n_cls*3)]
    accu_a = K.cast(
        K.equal(K.argmax(true_a), K.argmax(output_a)), K.floatx())
    accu_p = K.cast(
        K.equal(K.argmax(true_p), K.argmax(output_p)), K.floatx())
    accu_n = K.cast(
        K.equal(K.argmax(true_n), K.argmax(output_n)), K.floatx())
    return K.mean((accu_a+accu_p+accu_n)/3.0)


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


def discretize_with_histogram(tensor, bins):
    _min = K.min(tensor)
    _max = K.max(tensor)
    _len_shape = len(tensor.shape)
    _bins = K.cast(tf.range(bins), dtype=K.floatx())
    _range = tf.linspace(_min, _max, bins+1)
    for _ in range(_len_shape):
        _bins = K.expand_dims(_bins, axis=-1)
        _range = K.expand_dims(_range, axis=-1)
    _cond1 = K.greater_equal(tensor, _range[:-1])
    _cond2 = K.less(tensor, _range[1:])
    _cond3 = K.less_equal(tensor, _range[1:])
    _cond4 = K.concatenate((_cond2[:-1], _cond3[-1:]), axis=0)
    _all_cond = K.cast(K.all(K.stack((_cond1, _cond4), axis=0), axis=0),
                       dtype=K.floatx())
    _axis = tuple([i+1 for i in range(_len_shape)])
    _discrete = K.sum(_all_cond * _bins, axis=0)
    _histogram = tf.count_nonzero(_all_cond, axis=_axis)
    return _discrete, _histogram


def joint_histogram(tensor, bins):
    _uniq_obj = np.zeros((bins, bins, 2, 1))
    for i in range(bins):
        for j in range(bins):
            _uniq_obj[i, j, :, 0] = np.array([i, j])
    _uniq_obj = K.constant(_uniq_obj)
    _cond = K.all(K.equal(tensor, _uniq_obj), axis=2)
    return tf.count_nonzero(_cond, axis=2)


def mutual_information(tensor_a, tensor_b, bins=256):
    channel = tensor_a.shape[-1].value
    _a, _ = discretize_with_histogram(K.transpose(tensor_a)[i], bins=bins)
    _b, _ = discretize_with_histogram(K.transpose(tensor_b)[i], bins=bins)
    ab = K.stack([K.flatten(_a), K.flatten(_b)])
    joint_hist = K.cast(joint_histogram(ab, bins=bins), dtype=K.floatx())
    joint_proba = joint_hist/K.sum(joint_hist)
    joint_proba = K.clip(joint_proba, 1e-7, 1)
    a_proba = K.sum(joint_proba, axis=1)
    b_proba = K.sum(joint_proba, axis=0)
    a_proba = K.expand_dims(a_proba, axis=-1)
    mui = K.sum(joint_hist * joint_proba *
                K.log(joint_proba / (a_proba*b_proba)))
    return mui
