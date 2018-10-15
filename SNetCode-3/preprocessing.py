import math

# %% Preprocessing Module


def preproc(dataset):
    dataset = scaling(dataset)
    dataset = reshaping(dataset)
    return dataset


def scaling(dataset):
    mode = dataset.preprocessing
    if mode == None or mode == 'None':
        return dataset
    elif mode == '255' or mode == 255:
        dataset.X_train /= 255
        dataset.X_test /= 255
    else:
        print('\nError: Wrong Preprocessing Mode')
        quit()
    return dataset


def reshaping(dataset):
    shape = dataset.shape
    flat = shape[0]
    channels = shape[1]
    img_rows = shape[2] if len(shape) > 2 else None
    img_cols = shape[3] if len(shape) > 3 else None
    if (img_rows == None) and (img_cols == None):
        img_rows = img_cols = int(
            math.sqrt(dataset.X_train.shape[1] / channels))
    else:
        print('\nError: Wrong Reshape Inputs')
        quit()
    if flat == True:
        input_shape = dataset.X_train.shape[1]
    else:
        dataset.X_train = dataset.X_train.reshape(
            dataset.X_train.shape[0], img_rows, img_cols, channels)
        dataset.X_test = dataset.X_test.reshape(
            dataset.X_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)
    dataset.computed_input_shape = input_shape
    return dataset
