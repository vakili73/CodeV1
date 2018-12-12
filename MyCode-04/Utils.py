import os
import _pickle
import itertools
import numpy as np

from scipy import interp

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import TruncatedSVD, PCA

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras.utils import to_categorical


# %% Utils function

def plot_pca_reduction(embeddings, targets, title, save=True, figsize=(19.20, 10.80),
                       base_path='./logs/vizplots') -> plt.Figure:
    X = PCA(n_components=3).fit_transform(embeddings)
    plt.clf()
    plt.gcf().set_size_inches(*figsize)
    fig = plt.gcf()
    ax = Axes3D(fig)
    for i in range(len(np.unique(targets))):
        ind = np.where(targets == i)[0]
        ax.scatter(X[ind, 0], X[ind, 1], X[ind, 2])
    plt.title(title)
    if save:
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+title+'_pca.png'
        plt.savefig(path)
    return plt.gcf()


def plot_lsa_reduction(embeddings, targets, title, save=True, figsize=(19.20, 10.80),
                       base_path='./logs/vizplots') -> plt.Figure:
    X = TruncatedSVD(n_components=3).fit_transform(embeddings)
    plt.clf()
    plt.gcf().set_size_inches(*figsize)
    fig = plt.gcf()
    ax = Axes3D(fig)
    for i in range(len(np.unique(targets))):
        ind = np.where(targets == i)[0]
        ax.scatter(X[ind, 0], X[ind, 1], X[ind, 2])
    plt.title(title)
    if save:
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+title+'_lsa.png'
        plt.savefig(path)
    return plt.gcf()


def plot_reduction(**kwargs):
    embeddings = kwargs['embeddings']
    if isinstance(embeddings, list):
        targets = kwargs['targets']
        title = kwargs['title']
        for i in range(len(embeddings)):
            plot_lsa_reduction(embeddings[i], targets, title+'_'+str(i))
            plot_pca_reduction(embeddings[i], targets, title+'_'+str(i))
    else:
        plot_lsa_reduction(**kwargs)
        plot_pca_reduction(**kwargs)


def plot_lr_curve(history, title, save=True, figsize=(19.20, 10.80),
                  base_path='./logs/lrcurves') -> plt.Figure:
    plt.clf()
    plt.gcf().set_size_inches(*figsize)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.plot(history['epoch'],
             history['loss'],
             label='Train Loss')
    plt.plot(history['epoch'],
             history['val_loss'],
             label='Valid loss')
    plt.legend()
    if save:
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        path = base_path+'/'+title+'.png'
        plt.gcf().savefig(path)
    return plt.gcf()


def plot_roc_curve(title, y_test, y_score, n_cls, save=True, figsize=(19.20, 10.80),
                   base_path='./logs/roccurves') -> plt.Figure:
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_cls):
        fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score .ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_cls)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_cls):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_cls

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.clf()
    plt.gcf().set_size_inches(*figsize)
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
                          base_path='./logs/cmcurves', figsize=(19.20, 10.80)) -> plt.Figure:
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.clf()
    plt.gcf().set_size_inches(*figsize)
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


def load_loss(loss: str, **kwargs):
    if loss.startswith('K-'):
        loss = getattr(losses, loss[2:])
    elif loss.startswith('L-'):
        module = __import__('Losses')
        loss = getattr(module, loss[2:])(**kwargs)
    return loss


def load_datagen(datagen: str):
    if datagen == 'Original':
        return 'Original'
    module = __import__('Generator')
    datagen = getattr(module, datagen)
    return datagen


def load_metrics(metric: list, **kwargs):
    _metrics = []
    for m in metric:
        if m.startswith('K-'):
            _metrics.append(m[2:])
        elif m.startswith('L-'):
            module = __import__('Metrics')
            _metrics.append(getattr(module, m[2:])(**kwargs))
    return _metrics


def save_history(history, title,
                 base_path='./logs/histories'):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    path = base_path+'/'+title+'.cpkl'
    with open(path, 'wb') as fileObj:
        _pickle.dump(history, fileObj)
