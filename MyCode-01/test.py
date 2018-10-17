from Loader import getDataset
from Utils import preProcessing
from Generator import General
from Generator.Utils import Allo

db = getDataset('mnist')
db = preProcessing(db)

AugmentImageV2 = [Allo.CutOut, Allo.RndCrop,
                  Allo.RndRotate, Allo.Translate]

generator = General(X_train=db.X_train, y_train=db.y_train, augment=True, allowable=AugmentImageV2)
generator.get_batch()