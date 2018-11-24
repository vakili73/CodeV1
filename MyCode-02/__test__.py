from Config import KNN
from Database import INFO


from Database.Utils import reshape
from Database.Utils import load_data
from Database.Utils import get_fewshot

from Generator import MyTriplet
from sklearn.utils import shuffle
from Schema.Utils import load_schema

from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping

from Runs import MethodKNN
from Runs import report_classification

from Reporter import Reporter


# %% Initialization
way = -1
shot = None

s_ver = 'V01'
build = 'MyModelV2'

db = 'mnist'
n_cls = INFO[db]['n_cls']
shape = INFO[db]['shape']

rpt = Reporter(file_dir='./my_report.log')

# %% Dataset loading
data = load_data(db)

X_train, X_test, y_train, y_test = get_fewshot(*data, shot, way)

X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

X_train = reshape(X_train / 255.0, shape)
X_test = reshape(X_test / 255.0, shape)

# %% Generator section
traingen = MyTriplet(X_train, y_train, n_cls)
validgen = MyTriplet(X_test, y_test, n_cls)

# %% Schema creation
schema = load_schema(s_ver)
getattr(schema, 'build'+build)(shape, n_cls)

schema.model.compile(loss=[losses.categorical_crossentropy,
                           losses.kullback_leibler_divergence,
                           losses.kullback_leibler_divergence],
                     optimizer='adadelta', metrics=['acc'])

history = schema.model.fit_generator(
    traingen, epochs=1000, validation_data=validgen,
    callbacks=[EarlyStopping(patience=50)],
    workers=8, use_multiprocessing=True,
    verbose=2)
plot_lr_curve(history, 'my_history')

y_score = schema.myModel.predict(X_train)
report_classification(y_train, y_score, n_cls, 'my_train')
y_score = schema.myModel.predict(X_test)
report_classification(y_test, y_score, n_cls, 'my_test')

embed_train = schema.getModel().predict(X_train)
embed_test = schema.getModel().predict(X_test)

for weights, n_neighbors in KNN:
    MethodKNN(rpt, embed_train, embed_test, y_train, y_test,
              n_cls, weights, n_neighbors, 8, 'my_knn')
rpt.end_line()
