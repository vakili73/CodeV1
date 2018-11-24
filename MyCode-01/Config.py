
from Generator.Utils import Allo

AugmentImageV1 = [Allo.Correction,
                  Allo.CutOut, Allo.FlipLR,
                  Allo.PermCH, Allo.RndCrop,
                  Allo.RndRotate, Allo.Translate]

AugmentImageV2 = [Allo.CutOut, Allo.RndCrop,
                  Allo.RndRotate, Allo.Translate]

AugmentImageV3 = [Allo.Correction, Allo.CutOut,
                  Allo.PermCH, Allo.RndCrop,
                  Allo.RndRotate, Allo.Translate]

AugmentImageV4 = [Allo.Correction, Allo.CutOut,
                  Allo.FlipLR, Allo.RndCrop,
                  Allo.RndRotate, Allo.Translate]

Conf = [
    # dataset, model, augmented
    # ('cifar10', 'V07', AugmentImageV1),
    ('cifar100', 'V07', AugmentImageV1),
    ('fashion', 'V06', AugmentImageV4),
    # ('gisette', 'V03', []),
    # ('homus', 'V05', AugmentImageV2),
    # ('letter', 'V02', []),
    # ('mnist', 'V01', AugmentImageV2),
    # ('nist', 'V05', AugmentImageV2),
    # # ('omniglot', 'V09', AugmentImageV2), Old
    # ('omniglot', 'V10', AugmentImageV2),
    # ('pendigits', 'V02', []),
    # ('stl10', 'V08', AugmentImageV1),
    # ('svhn', 'V07', AugmentImageV3),
]


class Estm(object):
    Conventional = 'Conventional'
    Siamese = 'Siamese'
    Triplet = 'Triplet'
