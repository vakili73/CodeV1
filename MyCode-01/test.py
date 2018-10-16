from Loader import getDataset
from Utils import preProcessing
from Generator import Utils

db = getDataset('mnist')
db = preProcessing(db)

from matplotlib import pyplot as plt

img = db.X_train[0][:, :, 0]

def plot(img, title):
    plt.figure()
    plt.imshow(img)
    plt.title(title)

plot(img, 'Origin')

plot(Utils.FlipLR(img.copy()), Utils.Allo.FlipLR)
plot(Utils.FlipUD(img.copy()), Utils.Allo.FlipUD)
plot(Utils.PermCH(img.copy()), Utils.Allo.PermCH)
plot(Utils.CutOut(img.copy()), Utils.Allo.CutOut)
plot(Utils.RndCrop(img.copy()), Utils.Allo.RndCrop)
plot(Utils.RndRotate(img.copy()), Utils.Allo.RndRotate)
plot(Utils.Translate(img.copy()), Utils.Allo.Translate)
plot(Utils.Correction(img.copy()), Utils.Allo.Correction)

plt.show()