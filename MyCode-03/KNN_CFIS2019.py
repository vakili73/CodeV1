from Reporter import Reporter

from Database.Utils import load_data
from Database.Utils import get_fewshot

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":

    rpt = Reporter()

    datasets = [
        'stl10',
        'svhn',
        'mnist',
        'fashion',
        'cifar10',
    ]

    shots = [5, 15, None]

    for db in datasets:
        data = load_data(db)

        for shot in shots:
            print(db + ' ' + str(shot))
            X_train, X_test, y_train, y_test = get_fewshot(*data, shot)
            X_train, y_train = shuffle(X_train, y_train)
            X_test, y_test = shuffle(X_test, y_test)
            X_train = X_train / 255.0
            X_test = X_test / 255.0

            clf = KNeighborsClassifier(n_jobs=8)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accu = accuracy_score(y_test, y_pred)

            rpt.write_dataset(db)
            rpt.write_shot(shot)
            rpt.write_accuracy(accu)
            rpt.flush()
            rpt.end_line()

    rpt.close()
