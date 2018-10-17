from Database import Dataset

import os
import numpy as np

from scipy import interp
from matplotlib import pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


def rocCurve(title, y_test, y_score, n_cls, base_path='./logs/roccurves'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score [i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_cls)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_cls):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_cls

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = base_path+'/'+title+'.png'
    plt.savefig(path)


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
    if flat:
        return db
    img_rows = shape[1] if len(shape) > 2 else None
    img_cols = shape[2] if len(shape) > 3 else None
    channels = shape[3]
    db.X_train = db.X_train.reshape(
        db.X_train.shape[0], img_rows, img_cols, channels)
    db.X_test = db.X_test.reshape(
        db.X_test.shape[0], img_rows, img_cols, channels)
    return db
