import numpy as np
# write to work with 3D array


def _discrete_mutual_information(a, b, bins=2):

    if len(a.shape) > 2:
        channel = a.shape[0]
    else:
        a = np.expand_dims(a, axis=-1)
        b = np.expand_dims(b, axis=-1)
        channel = 1

    a = a.transpose()
    b = b.transpose()

    joint_histogram = np.zeros((bins, bins, channel))
    ab = np.stack([a, b], axis=-1)
    for c in range(channel):
        for i in range(bins):
            for j in range(bins):
                bcast = np.broadcast_to(
                    np.array([i, j]), ab[:, :, c, :].shape)
                joint_histogram[i, j, c] = np.sum(
                    np.all(np.equal(ab[:, :, c, :], bcast), axis=-1))

    joint_probability = joint_histogram / \
        np.sum(joint_histogram, axis=(0, 1))
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

    print(_discrete_mutual_information(image1, image2))
    print(_discrete_mutual_information(image1, image1.copy()))

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
