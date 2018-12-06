from Generator import MyTriplet

from Database import load_data
from Database.Utils import reshape


X_train, X_test, y_train, y_test = load_data('mnist')
X_train = reshape(X_train, (28, 28, 1))/255
X_test = reshape(X_test, (28, 28, 1))/255

