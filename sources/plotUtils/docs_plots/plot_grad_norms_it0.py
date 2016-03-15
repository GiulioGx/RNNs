import sys

from ActivationFunction import Tanh
from Configs import Configs
from descentDirectionRule.Antigradient import Antigradient
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.SpectralInit import SpectralInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer
from model.RNNManager import RNNManager
from output_fncs.Softmax import Softmax
from plotUtils.plt_utils import save_multiple_formats
from datasets.Dataset import InfiniteDataset
from datasets.TemporalOrderTask import TemporalOrderTask
from training.SGDTrainer import SGDTrainer
from training.TrainingRule import TrainingRule
from updateRule.SimpleUpdate import SimpleUdpate
import matplotlib.pyplot as plt

__author__ = 'giulio'

"""This script plots the gradient temporal norms for the first batch varying the spectral radius of W_rec"""

seed = 13
Configs.seed = seed


def __step(rho):
    print('Computing stats for rho: {} ...'.format(rho))
    # network setup
    std_dev = 0.1
    mean = 0
    vars_initializer = RNNVarsInitializer(
        W_rec_init=SpectralInit(matrix_init=GaussianInit(seed=seed, std_dev=std_dev), rho=rho),
        W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
        W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
        b_out_init=ConstantInit(0))
    net_initializer = RNNInitializer(vars_initializer, n_hidden=50)
    net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(),
                             output_fnc=Softmax())
    task = TemporalOrderTask(200, seed)
    out_dir = Configs.output_dir + str(task)
    loss_fnc = FullCrossEntropy(single_probability_ouput=False)
    dir_rule = Antigradient()
    lr_rule = GradientClipping(lr_value=0.001, clip_thr=1, normalize_wrt_dimension=False)  # 0.01
    update_rule = SimpleUdpate()
    train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)
    dataset = InfiniteDataset(task=task, validation_size=10 ** 4, n_batches=5)
    trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=5,
                         monitor_update_freq=200, batch_size=100)
    _, stats = trainer.train(dataset, net_builder)
    return stats


rho_values = [0.8, 0.9, 1, 1.1, 1.2]
n_plots = len(rho_values)
fig, axarr = plt.subplots(n_plots, sharex=True, sharey=False, figsize=(20, 30))

for i in range(n_plots):
    r = rho_values[i]
    stats = __step(r)
    y = stats.dictionary['obj_separate_norms'][0]['full_grad']
    ax = axarr[i]
    ax.bar(range(len(y)), y)
    ax.legend(['rho={}'.format(r)], shadow=True, fancybox=True)
    ax.set_yscale('log')
    # axarr[i].set_xlabel('t')
    # axarr[i].set_ylabel('temporal_grad_norm')

filename = sys.argv[0]
save_multiple_formats(filename)

plt.show()
