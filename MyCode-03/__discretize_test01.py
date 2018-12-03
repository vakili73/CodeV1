import numpy as np


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


def mutual_information(a, b, bins=2):
    a = a.transpose()
    b = b.transpose()

    mui = []
    for i in range(a.shape[2]):
        _a, _ = discretize_with_histogram(a.T[i], bins=bins)
        _b, _ = discretize_with_histogram(b.T[i], bins=bins)
        ab = np.stack([_a.flatten(), _b.flatten()])
        joint_hist = joint_histogram(ab, bins=bins)
        joint_proba = joint_hist/np.sum(joint_hist)
        joint_proba = np.clip(joint_proba, 1e-7, 1)
        a_proba = np.sum(joint_proba, axis=1)
        b_proba = np.sum(joint_proba, axis=0)
        a_proba = np.expand_dims(a_proba, axis=-1)
        mui.append(np.sum(joint_hist * joint_proba *
                          np.log(joint_proba / (a_proba*b_proba))))
    return np.mean(mui)


if __name__ == "__main__":

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

    print(mutual_information(image1, image2))
    print(mutual_information(image1, image1.copy()))

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

    print(mutual_information(image1, image2))
