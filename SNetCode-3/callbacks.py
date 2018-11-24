# pylint: disable=W0614
from tensorflow.keras.callbacks import * # noqa

def early_stopping():
    return EarlyStopping(monitor='val_loss', patience=10)

def tensor_board():
    return TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32)