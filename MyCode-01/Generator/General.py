from .BaseGenerator import BaseGenerator
from . import Utils

import numpy as np


class General(BaseGenerator):
    def __init__(self, augment=False, allowable=[],
                 batch_size: int = 128,
                 X_train: np.ndarray = np.array([]),
                 y_train: np.ndarray = np.array([])):
        super().__init__(augment, allowable,
                         batch_size, X_train, y_train)
        pass

    def _shuffle(self):
        ind = np.random.permutation(self.length)
        self.X_train = self.X_train[ind]
        self.y_train = self.y_train[ind]

    def _check(self):
        if self.state == None:
            self.state = {}
            self._shuffle()
            self.state['step'] = 0
            self.state['steps_size'] = self.length//self.batch_size
        else:
            step = self.state['step']
            steps_size = self.state['steps_size']
            if step == steps_size:
                self._shuffle()
                self.state['step'] = 0
            else:
                self.state['step'] += 1

    def _augmented_batch(self):
        X_data = []
        y_data = []
        step = self.state['step']
        start = step * self.batch_size
        stop = (step+1) * self.batch_size
        if stop > self.length:
            stop = self.length
        for i in range(start, stop):
            rnd = np.random.rand()
            if rnd < 1/(len(self.allowable)+1):
                X_data.append(self.X_train[i])
                y_data.append(self.y_train[i])
            else:
                augmentor = np.random.choice(self.allowable)
                augmentor = getattr(Utils, augmentor)
                img = self.X_train[i].copy()
                ch = img.shape[2]
                if ch == 1:
                    img = np.reshape(img, img.shape[0:2])
                    img = augmentor(img)
                    img = np.reshape(img, (*img.shape, ch))
                else:
                    img = augmentor(img)
                X_data.append(img)
                y_data.append(self.y_train[i])
        return np.array(X_data), np.array(y_data)

    def _nonaugmented_batch(self):
        X_data = []
        y_data = []
        step = self.state['step']
        start = step * self.batch_size
        stop = (step+1) * self.batch_size
        if stop > self.length:
            stop = self.length
        for i in range(start, stop):
            X_data.append(self.X_train[i])
            y_data.append(self.y_train[i])
        return np.array(X_data), np.array(y_data)

    def get_batch(self):
        self._check()
        if self.augment:
            return self._augmented_batch()
        else:
            return self._nonaugmented_batch()
        pass
