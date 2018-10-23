from sklearn.utils import shuffle
from Database.Utils import load_data, get_fewshot


data = load_data('mnist')

X_train, X_test, y_train, y_test = get_fewshot(*data, 1)
X_train, X_test, y_train, y_test = get_fewshot(*data, 5)
X_train, X_test, y_train, y_test = get_fewshot(*data, 5, 5)

X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)