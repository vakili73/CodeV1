
from Database import Dataset

from sklearn.metrics import classification_report

def classificationReport(title, y_true, y_pred):
    print(title)
    return classification_report(y_true, y_pred, digits=5)


def preProcessing(db: Dataset) -> Dataset:
    db = scaling(db)
    db = reshaping(db)
    return db


def scaling(db: Dataset) -> Dataset:
    mode = db.info['preproc']
    if mode == None or mode == 'None':
        return db
    elif mode == '255' or mode == 255:
        db.X_train /= 255
        db.X_test /= 255
    return db


def reshaping(db: Dataset) -> Dataset:
    shape = db.info['shape']
    flat = shape[0]
    img_rows = shape[1] if len(shape) > 2 else None
    img_cols = shape[2] if len(shape) > 3 else None
    channels = shape[3]
    if flat != True:
        db.X_train = db.X_train.reshape(
            db.X_train.shape[0], img_rows, img_cols, channels)
        db.X_test = db.X_test.reshape(
            db.X_test.shape[0], img_rows, img_cols, channels)
    return db
