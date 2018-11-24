
Info = {
    # shape: (--flat, rows, cols, channel)
    "cifar10": {
        "n_cls": 10,
        "preproc": 255,
        "shape": (False, 32, 32, 3),
    },
    "cifar100": {
        "n_cls": 100,
        "preproc": 255,
        "shape": (False, 32, 32, 3),
    },
    "fashion": {
        "n_cls": 10,
        "preproc": 255,
        "shape": (False, 28, 28, 1),
    },
    "gisette": {
        "n_cls": 2,
        "preproc": None,
        "shape": (True, 5000,),
    },
    "homus": {
        "n_cls": 32,
        "preproc": 255,
        "shape": (False, 40, 40, 1),
    },
    "letter": {
        "n_cls": 26,
        "preproc": None,
        "shape": (True, 16,),
    },
    "mnist": {
        "n_cls": 10,
        "preproc": 255,
        "shape": (False, 28, 28, 1),
    },
    "nist": {
        "n_cls": 26,
        "preproc": 255,
        "shape": (False, 32, 32, 1),
    },
    "pendigits": {
        "n_cls": 10,
        "preproc": None,
        "shape": (True, 16,),
    },
    "satimage": {
        "n_cls": 6,
        "preproc": None,
        "shape": (False, 3, 3, 4),
    },
    "stl10": {
        "n_cls": 10,
        "preproc": 255,
        "shape": (False, 96, 96, 3),
    },
    "svhn": {
        "n_cls": 10,
        "preproc": 255,
        "shape": (False, 32, 32, 3),
    },
    "usps": {
        "n_cls": 10,
        "preproc": None,
        "shape": (False, 16, 16, 1),
    },
}
