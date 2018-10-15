import os
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, alg, db, sim_num):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label='Valid loss')
    plt.ylim([0, 3])
    plt.legend()
    path = './figures'
    if not os.path.exists(path):
        os.makedirs(path)
    path += '/'+alg+'_'+db+'_'+str(sim_num)+'.png'
    plt.savefig(path)
