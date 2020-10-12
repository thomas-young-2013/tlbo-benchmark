import os
import sys
import argparse
import numpy as np
import pickle

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='caltech256')
parser.add_argument('--mode', choices=['local', 'server'], default='local')
args = parser.parse_args()

desktop_location = '/home/thomas/Desktop/codes/tl-benchmark'
if args.mode == 'local':
    sys.path.append(desktop_location)

from benchmark.utils.load_dataset import load_dataset
from benchmark.eval_functions.resnet.evaluate_resnet import train


def create_configspace():
    cs = ConfigurationSpace()
    lr = UniformFloatHyperparameter("learning_rate", 1e-2, 0.5, default_value=0.1, q=1e-2, log=True)
    momentum = UniformFloatHyperparameter("momentum", 1e-6, .95, default_value=.9)
    weight_decay = UniformFloatHyperparameter("weight_decay", 1e-5, 1e-3, default_value=1e-4, q=1e-5)
    nesterov = CategoricalHyperparameter("nesterov", [True, False], default_value=False)
    batch_size = UniformFloatHyperparameter("batch_size", 32, 256, q=16., default_value=128)
    padding_size = UniformFloatHyperparameter("padding_size", 1, 5, q=1., default_value=2)
    lr_reductions = UniformFloatHyperparameter("lr_decay_factor", 0.05, 0.5, default_value=0.1, q=0.05)
    cs.add_hyperparameters([lr, nesterov, momentum, lr_reductions, batch_size, weight_decay, padding_size])
    return cs


def get_default_setting(dataset_name):
    if dataset_name == 'cifar-10':
        return 165, 30
    elif dataset_name == 'cifar-100':
        return 200, 30
    elif dataset_name == 'svhn':
        return 50, 30
    elif dataset_name == 'tiny-imagenet':
        return 200, 30
    elif dataset_name == 'plant_seedling':
        return 50, 30
    elif dataset_name == 'dog_breed':
        return 150, 30
    elif dataset_name == 'dog_vs_cat':
        return 50, 30
    elif dataset_name == 'char74k':
        return 100, 30
    elif dataset_name == 'caltech101':
        return 100, 30
    elif dataset_name == 'caltech256':
        return 200, 30
    else:
        raise ValueError('Invalid dataset name!')


def main(dataset_name):
    (x_train, y_train), (x_test, y_test), cls_num = load_dataset(dataset_name)
    epochs_num, run_count = get_default_setting(dataset_name)

    # Scenario object
    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": run_count,
                         "cs": create_configspace(),
                         "deterministic": "true",
                         })

    def wrapper_for_resnet_tuning(cfg):
        val_error, test_error = train(cls_num, epochs_num, cfg, (x_train, y_train), (x_test, y_test), proportion=0.2, seed=32)
        print('the validation accuracy is ', 1 - val_error)
        print('the test accuracy is ', 1 - test_error)

        tmp_file = 'data/tmp_%s_result.pkl' % dataset_name
        if not os.path.exists(tmp_file):
            data = []
        else:
            with open(tmp_file, 'rb') as f:
                data = pickle.load(f)
        data.append((cfg.get_dictionary(), cfg.get_array(), val_error, test_error))

        with open(tmp_file, 'wb') as f:
            pickle.dump(data, f)
        return val_error

    smac = SMAC(scenario=scenario, rng=np.random.RandomState(60), tae_runner=wrapper_for_resnet_tuning)
    incumbent = smac.optimize()
    print('The best configuration is', incumbent)

    tmp_file = 'data/tmp_%s_result.pkl' % dataset_name
    target_file = 'data/result_%s_%d.pkl' % (dataset_name, run_count)
    os.rename(tmp_file, target_file)


if __name__ == "__main__":
    main(args.dataset)
