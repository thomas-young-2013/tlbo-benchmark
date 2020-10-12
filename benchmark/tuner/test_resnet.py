import os
import sys
import argparse
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, UnParametrizedHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace
from litebo.facade.bo_facade import BayesianOptimization as BO

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='caltech256')
parser.add_argument('--trial_num', type=int, default=500)
parser.add_argument('--mode', type=str, default='bo')
parser.add_argument('--gpu_device', type=int, default=0)

args = parser.parse_args()
dataset_str = args.datasets
mode = args.mode
trial_num = args.trial_num
gpu_device = 'cuda:%d' % args.gpu_device
data_dir = 'data/'

sys.path.append(os.getcwd())
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


def train(class_num, epoch_num, config, x_train, y_train, x_val, y_val, seed=32):
    epoch_num = int(epoch_num)
    print(epoch_num, config)

    train_batch_size = config['train_batch_size']
    init_lr = config['init_lr']
    lr_decay_factor = config['lr_decay_factor']
    weight_decay = config['weight_decay']
    momentum = config['momentum']
    nesterov = True if config['nesterov'] == 'True' else False

    from torchvision.models.resnet import resnet18
    model = resnet18(num_classes=class_num).to(gpu_device)

    x_train_data = torch.from_numpy(x_train)
    y_train_data = torch.from_numpy(y_train)
    train_dataset = TensorDataset(x_train_data, y_train_data)
    x_val_data = torch.from_numpy(x_val)
    y_val_data = torch.from_numpy(y_val)
    val_dataset = TensorDataset(x_val_data, y_val_data)

    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=5, shuffle=True)
    validloader = DataLoader(val_dataset, batch_size=100, num_workers=5, shuffle=False)

    optimizer = SGD(params=model.parameters(), lr=init_lr, momentum=momentum,
                    weight_decay=weight_decay, nesterov=nesterov)

    scheduler = MultiStepLR(optimizer, milestones=[int(epoch_num / 2), int(epoch_num * 3 / 4)],
                            gamma=lr_decay_factor)
    loss_func = nn.CrossEntropyLoss()

    for epoch_id in range(epoch_num):
        model.train()
        # print('Current learning rate: %.5f' % optimizer.state_dict()['param_groups'][0]['lr'])
        epoch_avg_loss = 0
        epoch_avg_acc = 0
        val_avg_loss = 0
        val_avg_acc = 0
        num_train_samples = 0
        num_val_samples = 0
        for i, data in enumerate(trainloader):
            batch_x, batch_y = data[0], data[1]
            num_train_samples += len(batch_x)
            logits = model(batch_x.float().to(gpu_device))
            loss = loss_func(logits, batch_y.to(gpu_device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_avg_loss += loss.to('cpu').detach() * len(batch_x)
            prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
            epoch_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

        epoch_avg_loss /= num_train_samples
        epoch_avg_acc /= num_train_samples

        print('Epoch %d: Train loss %.4f, train acc %.4f' % (epoch_id, epoch_avg_loss, epoch_avg_acc))

        if validloader is not None:
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(validloader):
                    batch_x, batch_y = data[0], data[1]
                    logits = model(batch_x.float().to(gpu_device))
                    val_loss = loss_func(logits, batch_y.to(gpu_device))
                    num_val_samples += len(batch_x)
                    val_avg_loss += val_loss.to('cpu').detach() * len(batch_x)

                    prediction = np.argmax(logits.to('cpu').detach().numpy(), axis=-1)
                    val_avg_acc += accuracy_score(prediction, batch_y.to('cpu').detach().numpy()) * len(batch_x)

                val_avg_loss /= num_val_samples
                val_avg_acc /= num_val_samples
                print('Epoch %d: Val loss %.4f, val acc %.4f' % (epoch_id, val_avg_loss, val_avg_acc))

        scheduler.step()
        return 1 - val_avg_acc


def run(dataset_name):
    file_id = '%s-resnet-%s-%d.pkl' % (dataset_name, mode, trial_num)
    saved_file = os.path.join(data_dir, file_id)

    def objective_function(cfg):
        (x_train, y_train), (x_test, y_test), cls_num = load_dataset(dataset_name)
        epochs_num, run_count = get_default_setting(dataset_name)
        val_error = train(cls_num, epochs_num, cfg, x_train, y_train, x_test, y_test, seed=32)
        print('the validation accuracy is ', 1 - val_error)

        if not os.path.exists(saved_file):
            data = list()
        else:
            with open(saved_file, 'rb') as f:
                data = pickle.load(f)
        data.append([cfg, val_error])

        with open(saved_file, 'wb') as f:
            pickle.dump(data, f)
        return val_error

    cs = create_configspace()
    bo = BO(objective_function, cs, max_runs=trial_num, time_limit_per_trial=10000,
            sample_strategy=mode, rng=np.random.RandomState(1))
    bo.run()


for dataset in dataset_str.split(','):
    run(dataset)
