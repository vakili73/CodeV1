
CVNAMES = ['A', 'B', 'C', 'D', 'E']
DATABASE = './Database'

INFO = {
    # shape: (rows, cols, channel)
    "cifar10": {
        "n_cls": 10,
        "shape": (32, 32, 3),
    },
    "cifar100": {
        "n_cls": 100,
        "shape": (32, 32, 3),
    },
    "fashion": {
        "n_cls": 10,
        "shape": (28, 28, 1),
    },
    "homus": {
        "n_cls": 32,
        "shape": (40, 40, 1),
    },
    "mnist": {
        "n_cls": 10,
        "shape": (28, 28, 1),
    },
    "nist": {
        "n_cls": 26,
        "shape": (32, 32, 1),
    },
    "omniglot": {
        "n_cls": 1623,
        "shape": (105, 105, 1),
    },
    "stl10": {
        "n_cls": 10,
        "shape": (96, 96, 3),
    },
    "svhn": {
        "n_cls": 10,
        "shape": (32, 32, 3),
    },
}
