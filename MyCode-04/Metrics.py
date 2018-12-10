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


def my_accu(n_cls, ll_len, e_len=128):
    def _my_accu(y_true, y_pred):
        out_len = ll_len*(e_len*3)
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
    return _my_accu


def kullback_leibler(tensor_a, tensor_b):
    tensor_a = K.clip(tensor_a, K.epsilon(), 1.0)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return K.sum(tensor_a * K.log(tensor_a / tensor_b), axis=-1)


def entropy(tensor):
    tensor = K.clip(tensor, K.epsilon(), 1.0)
    return -K.sum(tensor * K.log(tensor), axis=-1)


def cross_entropy(tensor_a, tensor_b):
    tensor_a = K.clip(tensor_a, K.epsilon(), 1.0)
    tensor_b = K.clip(tensor_b, K.epsilon(), 1.0)
    return -K.sum(tensor_a * K.log(tensor_b), axis=-1)


def cosine_similarity(tensor_a, tensor_b):
    tensor_a = K.l2_normalize(tensor_a, axis=-1)
    tensor_b = K.l2_normalize(tensor_b, axis=-1)
    return K.sum(tensor_a * tensor_b, axis=-1)


def cosine_distance(tensor_a, tensor_b):
    tensor_a = K.l2_normalize(tensor_a, axis=-1)
    tensor_b = K.l2_normalize(tensor_b, axis=-1)
    return 1.0-K.sum(tensor_a * tensor_b, axis=-1)


def squared_l2_distance(tensor_a, tensor_b):
    return K.sum(K.square(tensor_a - tensor_b), axis=-1)
