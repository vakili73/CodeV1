
METHOD = {
    # name, loss
    # K: keras, L: local
    'ConventionalV1': {
        'loss': 'K-categorical_crossentropy',
        'metrics': ['acc'],
        'datagen': 'Original',
        'classification': '',
    },
    'SiameseV1': {
        'loss': 'K-binary_crossentropy',
        'metrics': ['acc'],
        'datagen': 'SiameseV1',
    },
    'SiameseV2': {
        'loss': 'L-contrastive',
        'metrics': ['acc'],
        'datagen': 'SiameseV2',
    },
    'TripletV1': {
        'loss': 'K-mean_squared_error',
        'metrics': ['acc'],
        'datagen': 'Triplet',
    },
    'TripletV2': {
        'loss': 'L-triplet',
        'metrics': ['acc'],
        'datagen': 'Triplet',
    },
}

KNN = [
    # weights, n_neighbors
    ('uniform', 1),
    ('distance', 3),
    ('uniform', 5),
    ('distance', 7),
    # ('uniform', 9),
]

FEWSHOT = [
    # shot, way
    (None, -1),
    (5, -1),
    (15, -1),
    # (5, 5),
    # (15, 5),
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
    # ('cifar10', 'V03', _DATAGEN_OPT_COLORED_IMAGE),
    # ('cifar100', 'V03', _DATAGEN_OPT_COLORED_IMAGE),
    # ('fashion', 'V01', {
    #     'rotation_range': 15,
    #     'width_shift_range': 0.15,
    #     'height_shift_range': 0.15,
    #     'shear_range': 0.15,
    #     'zoom_range': 0.15,
    #     'horizontal_flip': True,
    # }),
    # ('homus', 'V02', _DATAGEN_OPT_BW_IMAGE),
    # ('mnist', 'V01', _DATAGEN_OPT_BW_IMAGE),
    # ('nist', 'V02', _DATAGEN_OPT_BW_IMAGE),
    ('omniglot', 'V04', _DATAGEN_OPT_BW_IMAGE),
    ('stl10', 'V04', _DATAGEN_OPT_COLORED_IMAGE),
    ('svhn', 'V03', _DATAGEN_OPT_COLORED_IMAGE),
]
