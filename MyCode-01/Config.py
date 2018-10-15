
import enum

class Estm(enum.Enum):
    Conventional = 'Conventional'
    Siamese = 'Siamese'
    Triplet = 'Triplet'

Conf = {
    # dataset, model
    ('cifar10', 'V06'),
    # ('cifar100', 'V07'),
    # ('fashion', 'V06'),
    # ('gisette', 'V03'),
    # ('homus', 'V05'),
    # ('letter', 'V02'),
    # ('mnist', 'V01'),
    # ('nist', 'V05'),
    # ('omniglot', 'V08'),
    # ('pendigits', 'V02'),
    # ('satimage', 'V04'),
    # ('stl10', 'V08'),
    # ('svhn', 'V07'),
    # ('usps', 'V01'),
}