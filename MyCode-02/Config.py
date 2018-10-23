
METHOD = [
    # name, loss, dgen
    ('CNNSIGM', 'L.', 'Original'),
    ('CNNRELU', 'K.', 'Original'),
    ('SIAMEV1', '', ''),
    ('SIAMEV2', '', ''),
    ('TRIPLV1', '', ''),
    ('TRIPLV2', '', ''),
]

KNN = [
    # weights, n_neighbors
    ('uniform', 1),
    ('uniform', 3),
    ('uniform', 5),
    ('uniform', 7),
    ('uniform', 9),
    ('distance', 1),
    ('distance', 3),
    ('distance', 5),
    ('distance', 7),
    ('distance', 9),
]

FEWSHOT = [
    # shot, way
    (None, -1),
    (1, -1),
    (5, -1),
    (15, -1),
    (1, 5),
    (5, 5),
    (15, 5),
    (1, 10),
    (5, 10),
    (15, 10),
]

_DATAGEN_OPT_COLORED_IMAGE = {
    'rotation_range': 15,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'shear_range': 0.15,
    'channel_shift_range': 0.15,
    'zoom_range': 0.15,
    'horizontal_flip': True,
}

_DATAGEN_OPT_BW_IMAGE = {
    'rotation_range': 15,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'shear_range': 0.15,
    'zoom_range': 0.15,
}

CONFIG = [
    # dataset, schema, dgen_opt
    ('cifar10', 'V04', _DATAGEN_OPT_COLORED_IMAGE),
    ('cifar100', 'V04', _DATAGEN_OPT_COLORED_IMAGE),
    ('fashion', 'V03', {
        'rotation_range': 15,
        'width_shift_range': 0.15,
        'height_shift_range': 0.15,
        'shear_range': 0.15,
        'zoom_range': 0.15,
        'horizontal_flip': True,
    }),
    ('homus', 'V02', _DATAGEN_OPT_BW_IMAGE),
    ('mnist', 'V01', _DATAGEN_OPT_BW_IMAGE),
    ('nist', 'V02', _DATAGEN_OPT_BW_IMAGE),
    ('omniglot', 'V06', _DATAGEN_OPT_BW_IMAGE),
    ('stl10', 'V05', _DATAGEN_OPT_COLORED_IMAGE),
    ('svhn', 'V04', _DATAGEN_OPT_COLORED_IMAGE),
]
