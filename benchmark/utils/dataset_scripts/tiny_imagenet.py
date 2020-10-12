import os
import numpy as np
from skimage.transform import resize


def load_tiny_imagenet():
    cls_num = 200
    dest_data_dir = 'data/tiny_imagenet/'

    x_train = np.load(dest_data_dir + 'x_train.npy')
    x_test = np.load(dest_data_dir + 'x_test.npy')

    print(x_train.shape, x_test.shape)
    y_train = np.load(dest_data_dir + 'y_train.npy')
    y_test = np.load(dest_data_dir + 'y_test.npy')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_test = y_test.reshape(-1)
    y_train = y_train.reshape(-1)
    return (x_train, y_train), (x_test, y_test), cls_num
