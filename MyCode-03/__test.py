import numpy as np
import tensorflow as tf
from keras import backend as K


def body(i):
    a = tf.constant(1, dtype=tf.int32)
    x.write(i, a)
    return tf.add(i, 1)


def condition(i):
    return tf.less(i, 128)


with tf.Session():
    x = tf.TensorArray(tf.int32, 128)
    i = tf.constant(0)
    tf.global_variables_initializer().run()
    _i = tf.while_loop(condition, body, [i])
    print(_i.eval())
    print(x.stack().eval())
