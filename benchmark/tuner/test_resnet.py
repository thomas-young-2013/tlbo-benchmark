import os
import argparse
import numpy as np
import pickle

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace
from litebo.facade.bo_facade import BayesianOptimization as BO

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type='caltech256')
parser.add_argument('--trial_num', type=int, default=500)
parser.add_argument('--mode', type=str, default='bo')

args = parser.parse_args()
dataset_str = args.datasets
run_count = args.n
mode = args.mode
trial_num = args.trial_num
data_dir = 'data/'
from benchmark.utils.load_dataset import load_dataset


def create_configspace():
    cs = ConfigurationSpace()
    batch_size = UniformIntegerHyperparameter("train_batch_size", 32, 256, default_value=64, q=8)
    init_lr = UniformFloatHyperparameter('init_lr', lower=1e-3, upper=0.3, default_value=0.1, log=True)
    lr_decay_factor = UnParametrizedHyperparameter('lr_decay_factor', 0.1)
    weight_decay = UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-2, default_value=0.0002,
                                              log=True)
    momentum = UniformFloatHyperparameter("momentum", 0.5, .99, default_value=0.9)
    nesterov = CategoricalHyperparameter('nesterov', ['True', 'False'], default_value='True')
    cs.add_hyperparameters([batch_size, init_lr, lr_decay_factor, weight_decay, momentum, nesterov])
    return cs


def get_default_setting(dataset_name):
    if dataset_name == 'cifar-10':
        return 165, trial_num
    elif dataset_name == 'cifar-100':
        return 200, trial_num
    elif dataset_name == 'svhn':
        return 50, trial_num
    elif dataset_name == 'tiny-imagenet':
        return 200, trial_num
    elif dataset_name == 'plant_seedling':
        return 50, trial_num
    elif dataset_name == 'dog_breed':
        return 150, trial_num
    elif dataset_name == 'dog_vs_cat':
        return 50, trial_num
    elif dataset_name == 'char74k':
        return 100, trial_num
    elif dataset_name == 'caltech101':
        return 100, trial_num
    elif dataset_name == 'caltech256':
        return 200, trial_num
    else:
        raise ValueError('Invalid dataset name!')


def run(dataset_name):
    file_id = '%s-resnet-%s-%d.pkl' % (dataset_name, mode, trial_num)
    saved_file = os.path.join(data_dir, file_id)

    def objective_function(cfg):
        (x_train, y_train), (x_test, y_test), cls_num = load_dataset(dataset_name)
        epochs_num, run_count = get_default_setting(dataset_name)
        val_error, test_error = train(cls_num, epochs_num, cfg, (x_train, y_train), (x_test, y_test), proportion=0.2, seed=32)
        print('the validation accuracy is ', 1 - val_error)
        print('the test accuracy is ', 1 - test_error)

        if not os.path.exists(saved_file):
            data = list()
        else:
            with open(saved_file, 'rb') as f:
                data = pickle.load(f)
        data.append([cfg, val_error, test_error])

        with open(saved_file, 'wb') as f:
            pickle.dump(data, f)
        return val_error

    cs = create_configspace()
    bo = BO(objective_function, cs, max_runs=trial_num, time_limit_per_trial=10000,
            sample_strategy=mode, rng=np.random.RandomState(1))
    bo.run()


for dataset in dataset_str.split(','):
    run(dataset)
