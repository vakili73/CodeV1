import numpy as np
# write to work with 1D array
from __discretize_test01 import discretize_with_histogram
from __discretize_test01 import joint_histogram as _joint_histogram


def _discrete_mutual_information(a, b, bins=2):
    a, _ = discretize_with_histogram(a, bins=bins)
    b, _ = discretize_with_histogram(b, bins=bins)

    ab = np.stack([a.flatten(), b.flatten()])
    joint_histogram = _joint_histogram(ab, bins=bins)
    joint_probability = joint_histogram/np.sum(joint_histogram)
    joint_probability = np.clip(joint_probability, 1e-7, 1)

    a_marginal_proba = np.sum(joint_probability, axis=1)
    b_marginal_proba = np.sum(joint_probability, axis=0)
    b_marginal_proba = np.expand_dims(b_marginal_proba, axis=-1)

    mui = joint_histogram * joint_probability *\
        np.log(joint_probability / (a_marginal_proba*b_marginal_proba))

    return np.sum(mui)


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

    print(_discrete_mutual_information(image1, image2))
    print(_discrete_mutual_information(image1, image1.copy()))

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

    print(_discrete_mutual_information(image1, image2))
