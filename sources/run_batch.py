import shutil
import sys

import theano

from ActivationFunction import Tanh
from Configs import Configs
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from model.RNNGrowingPolicy import RNNIncrementalGrowing
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer
from model.RNNManager import RNNManager
from output_fncs.Softmax import Softmax
from task.Dataset import InfiniteDataset
from task.TemporalOrderTask import TemporalOrderTask
from training.SGDTrainer import SGDTrainer
from training.TrainingRule import TrainingRule
from updateRule.SimpleUpdate import SimpleUdpate

__author__ = 'giulio'

separator = '#####################'

# ###THEANO CONFIG ### #
floatX = theano.config.floatX
device = theano.config.device
Configs.floatType = floatX
print(separator)
print('THEANO CONFIG')
print('device: ' + device)
print('floatType: ' + floatX)
print(separator)
# sys.setrecursionlimit(100000)


def train_run(seed: int, task_length: int, prefix: str):
    Configs.seed = seed

    # network setup
    mean = 0
    vars_initializer = RNNVarsInitializer(
        W_rec_init=SpectralInit(matrix_init=UniformInit(seed=seed), rho=1.2),
        W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
        W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
        b_out_init=ConstantInit(0))
    net_initializer = RNNInitializer(vars_initializer, n_hidden=5)
    net_growing_policy = RNNIncrementalGrowing(n_hidden_incr=5, n_hidden_max=50, n_hidden_incr_freq=500,
                                               initializer=vars_initializer)
    net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(), output_fnc=Softmax(),
                             growing_policy=net_growing_policy)

    loss_fnc = FullCrossEntropy(single_probability_ouput=False)

    # combining_rule = OnesCombination(normalize_components=False)
    combining_rule = SimplexCombination(normalize_components=True, seed=seed)
    # combining_rule = SimpleSum()
    dir_rule = CombinedGradients(combining_rule)

    lr_rule = GradientClipping(lr_value=0.001, clip_thr=1, normalize_wrt_dimension=False)  # 0.01

    # update_rule = FixedAveraging(t=10)
    update_rule = SimpleUdpate()
    # update_rule = Momentum(gamma=0.1)
    train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)

    task = TemporalOrderTask(task_length, seed)
    out_dir = Configs.output_dir + prefix + '/' + str(task) + '_' + str(seed)
    dataset = InfiniteDataset(task=task, validation_size=10 ** 4, n_batches=5)

    trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                         check_freq=200, batch_size=100, stop_error_thresh=1)
    net = trainer.train(dataset, net_builder)
    return net


seeds = [13, 14, 15, 16, 17]
lengths = [100]
prefix = 'train_run'

print('Beginning train run...')
print('seeds: {}, lengths: {}'.format(seeds, lengths))

shutil.rmtree(Configs.output_dir+prefix, ignore_errors=True)

for i in range(len(seeds)):
    for j in range(len(lengths)):
        train_run(seed=seeds[i], task_length=lengths[j], prefix=prefix)