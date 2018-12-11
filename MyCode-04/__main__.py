from Utils import load_loss
from Utils import load_datagen

from Database import INFORM

from Database import load_data
from Database import get_fewshot
from Database.Utils import reshape
from Database.Utils import plot_histogram

from Schema import load_schema

from Metrics import my_accu

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


schm = 'V03'
db = 'cifar10'
data = load_data(db)

# plot_histogram(data[-2], 'train_'+db)

X_train, X_test, y_train, y_test = get_fewshot(*data, shot=None)

X_train = reshape(X_train / 255.0, INFORM[db]['shape'])
X_test = reshape(X_test / 255.0, INFORM[db]['shape'])

schema = load_schema(schm).buildMyModelV2(
    INFORM[db]['shape'], INFORM[db]['n_cls'])

# schema.model.summary()

loss = load_loss('L-my_loss', n_cls=INFORM[db]['n_cls'], e_len=schema.e_len)
dgen = load_datagen('MyTriplet')

# schema.model.compile(optimizer='adadelta',
#                      loss='categorical_crossentropy', metrics=['acc'])

schema.model.compile(optimizer='adadelta', loss=loss,
                     metrics=[my_accu(INFORM[db]['n_cls'], schema.e_len)])

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.25)

# history = schema.model.fit(
#     X_train, to_categorical(y_train, INFORM[db]['n_cls']), epochs=1000,
#     batch_size=32, callbacks=[EarlyStopping(patience=20)], verbose=1,
#     validation_data=(X_valid, to_categorical(y_valid, INFORM[db]['n_cls'])))

traingen = dgen(X_train, y_train, INFORM[db]['n_cls'], 32)
validgen = dgen(X_valid, y_valid, INFORM[db]['n_cls'], 32)

history = schema.model.fit_generator(traingen, epochs=1000, validation_data=validgen,
                                     callbacks=[EarlyStopping(patience=20)],
                                     verbose=1, workers=8, use_multiprocessing=True)
