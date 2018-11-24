
from Schema import *
from Losses import *
from Estimator import *

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

# %% Initialization

runs = {
    # (Estimator, ModelSchema, Loss, Optimizer, Metric, Callback)
    # "cifar10": [
    #     (
    #         ConvenientNN,
    #         ModelSchemaV06_1I1O(),
    #         losses.categorical_crossentropy,
    #         optimizers.Adadelta(),
    #         [metrics.categorical_accuracy],
    #         [callbacks.EarlyStopping(patience=10)],),
    #     (
    #         SiameseDouble,
    #         ModelSchemaV06_2I1O(),
    #         contrastive_loss(),
    #         optimizers.Adadelta(),
    #         [], [],),
    #     (
    #         SiameseTriplet,
    #         ModelSchemaV06_3I3O(),
    #         triplet_loss(),
    #         optimizers.Adadelta(),
    #         [], [],),
    # ],
    # "cifar100": [
    #     (
    #         ConvenientNN,
    #         ModelSchemaV07_1I1O(),
    #         losses.categorical_crossentropy,
    #         optimizers.Adadelta(),
    #         [metrics.categorical_accuracy],
    #         [callbacks.EarlyStopping(patience=10)],),
    #     (
    #         SiameseDouble,
    #         ModelSchemaV07_2I1O(),
    #         contrastive_loss(),
    #         optimizers.Adadelta(),
    #         [], [],),
    #     (
    #         SiameseTriplet,
    #         ModelSchemaV07_3I3O(),
    #         triplet_loss(),
    #         optimizers.Adadelta(),
    #         [], [],),
    # ],
    "fashion": [
        # (
        #     ConvenientNN,
        #     ModelSchemaV06_1I1O(),
        #     losses.categorical_crossentropy,
        #     optimizers.Adadelta(),
        #     [metrics.categorical_accuracy],
        #     [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV06_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV06_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "gisette": [
        (
            ConvenientNN,
            ModelSchemaV03_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV03_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV03_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "homus": [
        (
            ConvenientNN,
            ModelSchemaV05_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV05_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV05_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "letter": [
        (
            ConvenientNN,
            ModelSchemaV02_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV02_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV02_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "mnist": [
        (
            ConvenientNN,
            ModelSchemaV01_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV01_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV01_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "nist": [
        (
            ConvenientNN,
            ModelSchemaV05_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV05_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV05_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "pendigits": [
        (
            ConvenientNN,
            ModelSchemaV02_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV02_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV02_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "satimage": [
        (
            ConvenientNN,
            ModelSchemaV04_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV04_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV04_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "stl10": [
        (
            ConvenientNN,
            ModelSchemaV08_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV08_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV08_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "svhn": [
        (
            ConvenientNN,
            ModelSchemaV07_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV07_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV07_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
    "usps": [
        (
            ConvenientNN,
            ModelSchemaV01_1I1O(),
            losses.categorical_crossentropy,
            optimizers.Adadelta(),
            [metrics.categorical_accuracy],
            [callbacks.EarlyStopping(patience=10)],),
        (
            SiameseDouble,
            ModelSchemaV01_2I1O(),
            contrastive_loss(),
            optimizers.Adadelta(),
            [], [],),
        (
            SiameseTriplet,
            ModelSchemaV01_3I3O(),
            triplet_loss(),
            optimizers.Adadelta(),
            [], [],),
    ],
}
