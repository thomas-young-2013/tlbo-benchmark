from keras.datasets import cifar10
from keras.datasets import cifar100
from benchmark.utils.dataset_scripts.svhn import load_svhn
from benchmark.utils.dataset_scripts.tiny_imagenet import load_tiny_imagenet
from benchmark.utils.dataset_scripts.plant_seedling import load_plant_seedling
from benchmark.utils.dataset_scripts.dog_breed import load_dog_breed
from benchmark.utils.dataset_scripts.dog_vs_cat import load_dog_vs_cat
from benchmark.utils.dataset_scripts.char74k import load_char74k
from benchmark.utils.dataset_scripts.caltech import load_caltech


def load_cifar_10():
    cls_num = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test), cls_num


def load_cifar_100():
    cls_num = 100
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return (x_train, y_train), (x_test, y_test), cls_num


def load_dataset(dataset_name='cifar-10'):
    if dataset_name == 'cifar-10':
        return load_cifar_10()
    elif dataset_name == 'cifar-100':
        return load_cifar_100()
    elif dataset_name == 'svhn':
        return load_svhn()
    elif dataset_name == 'tiny-imagenet':
        return load_tiny_imagenet()
    elif dataset_name == 'plant_seedling':
        return load_plant_seedling()
    elif dataset_name == 'dog_breed':
        return load_dog_breed()
    elif dataset_name == 'dog_vs_cat':
        return load_dog_vs_cat()
    elif dataset_name == 'char74k':
        return load_char74k()
    elif dataset_name == 'caltech101':
        return load_caltech(cls_num=101)
    elif dataset_name == 'caltech256':
        return load_caltech(cls_num=256)
    else:
        raise ValueError('undefined dataset name!')
