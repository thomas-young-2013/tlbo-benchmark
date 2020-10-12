import os
import numpy as np
from skimage.transform import resize


def load_tiny_imagenet():
    cls_num = 200
    source_data_dir = 'data/tiny_imagenet/Single30epoch2/'
    dest_data_dir = 'data/tiny_imagenet/data/'
    if not os.path.exists(dest_data_dir + 'x_train_32.npy'):
        x_train = np.load(source_data_dir + 'x_train.npy')
        x_train_tmp = list()
        for item in x_train:
            tmp = resize(item, (32, 32, 3), anti_aliasing=True)
            x_train_tmp.append(tmp)
        x_train = np.array(x_train_tmp)
        np.save(dest_data_dir+'x_train_32.npy', x_train)

        x_test = np.load(source_data_dir + 'x_test.npy')
        x_test_tmp = list()
        for item in x_test:
            tmp = resize(item, (32, 32, 3), anti_aliasing=True)
            x_test_tmp.append(tmp)
        x_test = np.array(x_test_tmp)
        np.save(dest_data_dir+'x_test_32.npy', x_test)
    else:
        x_train = np.load(dest_data_dir + 'x_train_32.npy')
        x_test = np.load(dest_data_dir + 'x_test_32.npy')

    print(x_train.shape, x_test.shape)
    y_train = np.load(source_data_dir + 'y_train.npy')
    y_test = np.load(source_data_dir + 'y_test.npy')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    y_test = y_test.reshape(-1)
    y_train = y_train.reshape(-1)
    return (x_train, y_train), (x_test, y_test), cls_num
