import os
import itertools
import numpy as np

from scipy import interp
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import TruncatedSVD


# %% Utils function

def plot_pca_reduction(embeddings, targets, title):
    X_pca = TruncatedSVD(n_components=2).fit_transform(embeddings)

    plt.figure()
    

def plot_lr_curve(history, title, ylim=(0, 3), save=True,
                  base_path='./logs/lrcurves') -> plt.Figure:
    plt.figure()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.plot(history.epoch,
             history.history['loss'],
             label='Train Loss')
    plt.plot(history.epoch,
             history.history['val_loss'],
             label='Valid loss')
    plt.ylim(ylim)
    plt.legend()
    if save:
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+title+'.png'
        plt.gcf().savefig(path)
    return plt.gcf()


def plot_roc_curve(title, y_test, y_score, n_cls, save=True,
                   base_path='./logs/roccurves') -> plt.Figure:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
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
    plt.title(title)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
             ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if save:
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+title+'.png'
        plt.savefig(path)
    return plt.gcf()


def plot_confusion_matrix(cm, title, classes, save=True, normalize=True,
                          base_path='./logs/cmcurves') -> plt.Figure:
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save:
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+title+'.png'
        plt.savefig(path)
    return plt.gcf()
