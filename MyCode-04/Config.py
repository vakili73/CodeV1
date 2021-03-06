
TOP_K_ACCU = [1, 3, 5]

__VERBOSE = 2
__EPOCHS = 700

PATIENCE = 10
BATCHSIZE = 32
OPTIMIZER = 'adam'
FITGENOPTS = {
    'workers': 8,
    'epochs': __EPOCHS,
    'verbose': __VERBOSE,
    'max_queue_size': 80,
    'steps_per_epoch': 1000,
    'validation_steps': 200,
}
FITOPTS = {
    'epochs': __EPOCHS,
    'verbose': __VERBOSE,
    'batch_size': BATCHSIZE,
}

__SHOTS = [5, 10, 20, 50, None]

__DATAGEN_OPT_COLORED_IMAGE = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'channel_shift_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
}

__DATAGEN_OPT_B_AND_W_IMAGE = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
}

DATASETS = {
    "cifar10": {
        "schema": 'V03',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
    "cifar100": {
        "schema": 'V03',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
    "fashion": {
        "schema": 'V01',
        "shots": __SHOTS,
        "dgen_opt": {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
        },
    },
    "homus": {
        "schema": 'V02',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    },
    "mingnet": {
        "schema": 'V04',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
    "mnist": {
        "schema": 'V01',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    },
    "nist": {
        "schema": 'V02',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    },
    "omniglot": {
        "schema": 'V01',
        "shots": [5, 10, None],
        "dgen_opt": __DATAGEN_OPT_B_AND_W_IMAGE,
    },
    "stl10": {
        "schema": 'V04',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
    "svhn": {
        "schema": 'V03',
        "shots": __SHOTS,
        "dgen_opt": __DATAGEN_OPT_COLORED_IMAGE,
    },
}

__N_JOBS = 8
__KNNS = [
    {'n_neighbors': 1,
     'weights': 'uniform',
     'n_jobs': __N_JOBS, },
    {'n_neighbors': 5,
     'weights': 'uniform',
     'n_jobs': __N_JOBS, },
    {'n_neighbors': 10,
     'weights': 'distance',
     'n_jobs': __N_JOBS, },
    {'n_neighbors': 15,
     'weights': 'distance',
     'n_jobs': __N_JOBS, },
]

__SVMS = [
    {'kernel': 'linear',
     'gamma': 'scale', },
    {'kernel': 'rbf',
     'gamma': 'scale', },
    # {'kernel': 'poly',
    #  'gamma': 'scale', },
    {'kernel': 'sigmoid',
     'gamma': 'scale', },
]

METHODS = {
    'ConventionalV1': {
        'loss': 'K-categorical_crossentropy',
        'metrics': ['K-acc'],
        'datagen': 'Original',
        'classification': '',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    # 'ConventionalV2': {
    #     'loss': 'K-categorical_crossentropy',
    #     'metrics': ['K-acc'],
    #     'datagen': 'Original',
    #     'classification': '',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    'MyModelV1': {
        'loss': 'L-my_loss',
        'metrics': ['L-my_accu'],
        'datagen': 'MyTriplet',
        'classification': '',
        'knn': (__KNNS, {'metric': ['cosine', 'kullbackleibler']}), # , 'correlation'
        'svm': __SVMS,
    },
    'MyModelV2': {
        'loss': 'L-my_loss',
        'metrics': ['L-my_accu'],
        'datagen': 'MyTriplet',
        'classification': '',
        'knn': (__KNNS, {'metric': ['cosine', 'kullbackleibler']}), # , 'correlation'
        'svm': __SVMS,
    },
    'SiameseV1': {
        'loss': 'K-binary_crossentropy',
        'metrics': [],
        'datagen': 'SiameseV1',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    'SiameseV2': {
        'loss': 'L-contrastive',
        'metrics': [],
        'datagen': 'SiameseV2',
        'knn': __KNNS,
        'svm': __SVMS,
    },
    # 'TripletV1': {
    #     'loss': 'K-mean_squared_error',
    #     'metrics': [],
    #     'datagen': 'Triplet',
    #     'knn': __KNNS,
    #     'svm': __SVMS,
    # },
    'TripletV2': {
        'loss': 'L-triplet',
        'metrics': [],
        'datagen': 'Triplet',
        'knn': __KNNS,
        'svm': __SVMS,
    },
}
