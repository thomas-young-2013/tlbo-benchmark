from scipy import io
import numpy as np


def load_svhn():
    cls_num = 10
    Train = io.loadmat('data/svhn/train_32x32.mat')
    Test = io.loadmat('data/svhn/test_32x32.mat')

    x_train = Train['X']
    y_train = Train['y']
    x_test = Test['X']
    y_test = Test['y']

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train[np.newaxis, ...]
    x_train = np.swapaxes(x_train, 0, 4).squeeze()

    x_test = x_test[np.newaxis, ...]
    x_test = np.swapaxes(x_test, 0, 4).squeeze()

    np.place(y_train, y_train == 10, 0)
    np.place(y_test, y_test == 10, 0)
    y_test = y_test.reshape(-1)
    y_train = y_train.reshape(-1)
    return (x_train, y_train), (x_test, y_test), cls_num
