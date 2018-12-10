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
from tensorflow.keras.callbacks import EarlyStopping


schm = 'V01'
db = 'fashion'
data = load_data(db)

plot_histogram(data[-2], 'train_'+db)

X_train, X_test, y_train, y_test = get_fewshot(*data, shot=None)

X_train = reshape(X_train / 255.0, INFORM[db]['shape'])
X_test = reshape(X_test / 255.0, INFORM[db]['shape'])

schema = load_schema(schm).buildMyModelV2(
    INFORM[db]['shape'], INFORM[db]['n_cls'])

loss = load_loss('L-my_loss', n_cls=INFORM[db]['n_cls'], ll_len=schema.ll_len)
dgen = load_datagen('MyTriplet')

schema.model.compile(optimizer='adam', loss=loss,
                     metrics=[my_accu(INFORM[db]['n_cls'], schema.ll_len)])

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.25)

traingen = dgen(X_train, y_train, INFORM[db]['n_cls'], 128)
validgen = dgen(X_train, y_train, INFORM[db]['n_cls'], 128)

history = schema.model.fit_generator(traingen, epochs=1000, validation_data=validgen,
                                     callbacks=[EarlyStopping(patience=10)],
                                     verbose=1, workers=8, use_multiprocessing=True)


