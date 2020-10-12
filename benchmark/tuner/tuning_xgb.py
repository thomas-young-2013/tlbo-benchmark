import os
import sys
import argparse
import numpy as np
import xgboost as xgb
import pickle

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='iris')
parser.add_argument('--max_run', type=int, default=100)
parser.add_argument('--mode', choices=['local', 'server'], default='local')
parser.add_argument('--nthread', type=int, default=4)
args = parser.parse_args()

if args.mode == 'local':
    sys.path.append('/home/changzhuo/codes/tl-benchmark')
    sys.path.append('/Users/changzhuo/codes/tl-benchmark')
    sys.path.append('/usr/codes/tl-benchmark')
if args.mode == 'server':
    sys.path.append('/home/hpt/tl-benchmark')
    sys.path.append('/home/liyang/codes/tl-benchmark')
    sys.path.append('/home/daim/thomas/tl-benchmark')

from benchmark.utils.load_xgb_dataset import load_data
parent_dir = 'data/xgb_metadata/'
run_count = args.max_run
dataset_name = args.dataset


def create_configspace():
    cs = ConfigurationSpace()

    n_estimators = UniformFloatHyperparameter("n_estimators", 100, 600, default_value=200, q=50)
    eta = UniformFloatHyperparameter("eta", 0.025, 0.3, default_value=0.3, q=0.025)
    min_child_weight = UniformIntegerHyperparameter("min_child_weight", 1, 10, default_value=1)
    max_depth = UniformIntegerHyperparameter("max_depth", 1, 14, default_value=6)
    subsample = UniformFloatHyperparameter("subsample", 0.5, 1, default_value=1, q=0.05)
    gamma = UniformFloatHyperparameter("gamma", 0, 1, default_value=0, q=0.1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.5, 1, default_value=1., q=0.05)
    alpha = UniformFloatHyperparameter("alpha", 0, 10, default_value=0., q=1.)
    _lambda = UniformFloatHyperparameter("lambda", 1, 2, default_value=1, q=0.1)
    scale_pos_weight = CategoricalHyperparameter("scale_pos_weight", [0.01, 0.1, 1., 10, 100], default_value=1.)

    cs.add_hyperparameters(
        [n_estimators, eta, min_child_weight, max_depth, subsample, gamma, colsample_bytree, alpha, _lambda,
         scale_pos_weight])
    return cs


# Output the tuning result.
def spill_result(item):
    tmp_file = parent_dir + ('%s_tmp_result.pkl' % dataset_name)
    if not os.path.exists(tmp_file):
        data = []
    else:
        with open(tmp_file, 'rb') as f:
            data = pickle.load(f)
    data.append(item)
    with open(tmp_file, 'wb') as f:
        pickle.dump(data, f)


# Rename the temporary file
def rename():
    tmp_file = parent_dir + ('%s_tmp_result.pkl' % dataset_name)
    target_file = parent_dir + ('result_%s_%d.pkl' % (dataset_name, run_count))
    os.rename(tmp_file, target_file)


def main():
    # Prepare the data for training.
    (x_train, y_train), (x_test, y_test), num_cls = load_data(dataset_name)
    dmtrain = xgb.DMatrix(x_train, label=y_train)
    dmvalid = xgb.DMatrix(x_test, label=y_test)
    cs = create_configspace()

    def objective_func(params):
        num_round = int(params['n_estimators'])
        parameters = {}
        for p in params:
            parameters[p] = params[p]

        if num_cls > 2:
            parameters['num_class'] = num_cls
            parameters['objective'] = 'multi:softmax'
            parameters['eval_metric'] = 'merror'
        elif num_cls == 2:
            parameters['objective'] = 'binary:logistic'
            parameters['eval_metric'] = 'error'

        parameters['tree_method'] = 'hist'
        parameters['booster'] = 'gbtree'
        parameters['nthread'] = args.nthread
        parameters['silent'] = 1
        watchlist = [(dmtrain, 'train'), (dmvalid, 'valid')]

        model = xgb.train(parameters, dmtrain, num_round, watchlist, verbose_eval=1)
        pred = model.predict(dmvalid)
        if num_cls == 2:
            pred = [int(i > 0.5) for i in pred]
        acc = accuracy_score(dmvalid.get_label(), pred)
        spill_result((params.get_dictionary(), params.get_array(), acc))
        return 1 - acc

    # Create scenario object
    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": run_count,
                         "cs": cs,
                         "deterministic": "true"
                         })

    smac = SMAC(scenario=scenario, rng=np.random.RandomState(60), tae_runner=objective_func)
    incumbent = smac.optimize()
    incumbent_value = objective_func(incumbent)
    print('The best configuration is', incumbent)
    print('The best performance, error:', incumbent_value)
    # Rename the temporary file.
    rename()


if __name__ == "__main__":
    main()
