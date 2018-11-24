import numpy as np

from tensorflow.keras import backend as K

from NpMetrics import mutual_information
from Metrics import mutual_information as tf_mutual_information

# binary and gray scale image mutual information

image1 = [
    [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0]
    ],
    [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0]
    ],
    [
        [0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0]
    ]
]

image2 = [
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ],
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ],
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]
]

image1 = np.array(image1)
image2 = np.array(image2)

t_image1 = K.variable(image1.transpose())
t_image2 = K.variable(image2.transpose())

# print(K.get_value(tf_mutual_information(t_image1, t_image2, bins=2)))
# print(K.get_value(tf_mutual_information(t_image1, t_image1, bins=2)))


def _discrete_mutual_information(a, b, bins=2, channel=3):

    a = a.transpose().astype('float')
    b = b.transpose().astype('float')

    _ab = np.stack([a, b], axis=-1)

    def _replace(x):
        new_x = np.zeros(x.shape, dtype='int')
        _max = np.max(x)
        _range = np.round(list(np.linspace(0, _max, bins+1)), decimals=6)
        for i in range(len(_range)-1):
            _cond1 = np.greater_equal(x, _range[i]-1e-7)
            if i+1 == len(_range)-1:
                _cond2 = np.less_equal(x, _range[i+1]+1e-7)
            else:
                _cond2 = np.less(x, _range[i+1]+1e-7)
            new_x[np.all(np.stack((_cond1, _cond2), axis=-1), axis=-1)] = i
        return new_x

    ab = np.zeros(_ab.shape, dtype='int')
    for c in range(channel):
        ab[:, :, c, :] = _replace(_ab[:, :, c, :])

    joint_histogram = np.zeros((bins, bins, channel), dtype='float')
    for c in range(channel):
        for i in range(bins):
            for j in range(bins):
                bcast = np.broadcast_to(np.array([i, j], dtype='int'),
                                        ab[:, :, c, :].shape)
                cond = np.equal(ab[:, :, c, :], bcast)
                joint_histogram[i, j, c] = np.sum(np.all(cond, axis=-1))

    joint_probability = joint_histogram/np.sum(joint_histogram, axis=(0, 1))
    joint_probability = np.clip(joint_probability, 1e-7, 1)

    a_marginal_proba = np.sum(joint_probability, axis=1)
    b_marginal_proba = np.sum(joint_probability, axis=0)

    mui = []
    for c in range(channel):
        _mui = []
        for i in range(bins):
            for j in range(bins):
                _mui.append(joint_histogram[i, j, c] * np.sum(
                    joint_probability[i, j, c] *
                    np.log(joint_probability[i, j, c] /
                           (a_marginal_proba[i, c]*b_marginal_proba[j, c]))))
        mui.append(np.sum(_mui))

    return np.mean(mui)


print(_discrete_mutual_information(image1, image2))
print(_discrete_mutual_information(image1, image1.copy()))

print(mutual_information(image1.transpose(), image2.transpose(), bins=2))
print(mutual_information(image1.transpose(), image1.copy().transpose(), bins=2))


image1 = [
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ],
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ],
    [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ],
]

image2 = [
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ],
]

image1 = np.array(image1)
image2 = np.array(image2)

print(_discrete_mutual_information(image1, image2))
print(mutual_information(image1.transpose(), image2.transpose(), bins=2))
