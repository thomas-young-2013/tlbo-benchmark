import numpy as np
from sklearn.model_selection import train_test_split


def load_char74k():
    cls_num = 62
    data_dir = 'data/char74k/'

    x_train = np.load(data_dir + 'x_train.npy')
    y_train = np.load(data_dir + 'y_train.npy')
    x_train = x_train.astype('float32')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=32,
                                                        stratify=y_train)
    y_test = y_test.reshape(-1)
    y_train = y_train.reshape(-1)
    return (x_train, y_train), (x_test, y_test), cls_num
