import numpy as np
# write to work with 1D array


def _continuous_mutual_information(a, b, bins=2):
    a = a.flatten()
    b = b.flatten()

    ab = np.stack([a, b], axis=-1)

    joint_histogram = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(bins):
            bcast = np.broadcast_to(np.array([i, j]), ab.shape)
            joint_histogram[i, j] = np.sum(
                np.all(np.equal(ab, bcast), axis=-1))

    joint_probability = joint_histogram/np.sum(joint_histogram)
    joint_probability = np.clip(joint_probability, 1e-7, 1)

    a_marginal_proba = np.sum(joint_probability, axis=1)
    b_marginal_proba = np.sum(joint_probability, axis=0)

    mui = 0
    for i in range(bins):
        for j in range(bins):
            mui += joint_histogram[i, j] * np.sum(
                joint_probability[i, j] *
                np.log(joint_probability[i, j] /
                       (a_marginal_proba[i]*b_marginal_proba[j])))
    return mui


if __name__ == "__main__":

    image1 = [[0, 0, 1, 0, 0],
              [0, 1, 0, 1, 0],
              [1, 0, 0, 0, 1],
              [0, 1, 0, 1, 0],
              [0, 0, 1, 0, 0]]

    image2 = [[0, 0, 1, 0, 0],
              [0, 1, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0]]

    image1 = np.array(image1)
    image2 = np.array(image2)

    print(_continuous_mutual_information(image1, image2))
    print(_continuous_mutual_information(image1, image1.copy()))

    image1 = [[1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1]]

    image2 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]

    image1 = np.array(image1)
    image2 = np.array(image2)

    print(_continuous_mutual_information(image1, image2))
