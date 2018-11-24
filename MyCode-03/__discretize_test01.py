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


if __name__ == "__main__":

    a = np.array([0.5, 0.5, 1., 1.5, 2., 2.5])
    b = np.array([
        [0., 0.5, 1., 1.5, 2., 2.5],
        [0., 0.5, 1., 1.5, 2., 2.5],
        [0., 0.5, 1., 1.5, 2., 2.5],
    ])
    c = np.array([
        [
            [0., 0.5, 1., 1.5, 2., 2.5],
            [0., 0.5, 1., 1.5, 2., 2.5],
            [0., 0.5, 1., 1.5, 2., 2.5],
        ],
        [
            [0., 0.5, 1., 1.5, 2., 2.5],
            [0., 0.5, 1., 1.5, 2., 2.5],
            [0., 0.5, 1., 1.5, 2., 2.5],
        ],
        [
            [0., 0.5, 1., 1.5, 2., 2.5],
            [0., 0.5, 1., 1.5, 2., 2.5],
            [0., 0.5, 1., 1.5, 2., 2.5],
        ],
    ])

    _min = np.min(a)
    _max = np.max(a)
    print(_discrete_with_histogram(a, 4))
    print(_discrete_with_histogram(b, 4))
    print(_discrete_with_histogram(c, 4))

    print(_obj_discrete_with_histogram(b))
