import sys

import theano

from ActivationFunction import Tanh
from Configs import Configs
from Paths import Paths
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.Antigradient import Antigradient
from descentDirectionRule.CombinedGradients import CombinedGradients
from descentDirectionRule.LBFGSDirection import LBFGSDirection
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.SVDInit import SVDInit
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from learningRule.GradientClipping import GradientClipping
from learningRule.ProbabilisticSearch import ProbabilisticSearch
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from metrics.BestValueFoundCriterion import BestValueFoundCriterion
from metrics.ErrorMonitor import ErrorMonitor
from metrics.LossMonitor import LossMonitor
from metrics.RocMonitor import RocMonitor
from metrics.ThresholdCriterion import ThresholdCriterion
from model.RNNGrowingPolicy import RNNIncrementalGrowing
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer
from model.RNNManager import RNNManager
from output_fncs.Logistic import Logistic
from output_fncs.Softmax import Softmax
from task.Dataset import InfiniteDataset
from task.LupusDataset import LupusDataset
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

seed = 15
Configs.seed = seed

# network setup
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
# vars_initializer = RNNVarsInitializer(
#     W_rec_init=SpectralInit(matrix_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed), rho=1.2),
#     W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
#     W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
#     b_out_init=ConstantInit(0))
vars_initializer = RNNVarsInitializer(
    W_rec_init=SpectralInit(matrix_init=GaussianInit(seed=seed, std_dev=std_dev), rho=1.2),
    W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
    W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
    b_out_init=ConstantInit(0))
net_initializer = RNNInitializer(vars_initializer, n_hidden=50)
net_growing_policy = RNNIncrementalGrowing(n_hidden_incr=5, n_hidden_max=50, n_hidden_incr_freq=1000,
                                           initializer=vars_initializer)
net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(),
                         output_fnc=Logistic())  # , growing_policy=net_growing_policy)

# setup
out_dir = Configs.output_dir + 'Lupus'
loss_fnc = FullCrossEntropy(single_probability_ouput=True)

# combining_rule = OnesCombination(normalize_components=False)
combining_rule = SimplexCombination(normalize_components=True, seed=seed)
# combining_rule = SimpleSum()
dir_rule = CombinedGradients(combining_rule)
# dir_rule = Antigradient()
# dir_rule = LBFGSDirection(n_pairs=7)

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
# lr_rule = ConstantNormalizedStep(0.001)  # 0.01
lr_rule = GradientClipping(lr_value=0.0001, clip_thr=1, normalize_wrt_dimension=False)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)

# update_rule = FixedAveraging(t=10)
update_rule = SimpleUdpate()
# update_rule = Momentum(gamma=0.1)


train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)

dataset = LupusDataset(mat_file=Paths.lupus_path)

loss_monitor = LossMonitor(loss_fnc=loss_fnc)
roc_monitor = RocMonitor()
#stopping_criterion = ThresholdCriterion(monitor=error_monitor, threshold=1. / 100)
#saving_criterion = BestValueFoundCriterion(monitor=error_monitor)

trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                     monitor_update_freq=50, batch_size=100)
trainer.add_monitors(dataset.train_set, 'train', loss_monitor, roc_monitor)
#trainer.set_saving_criterion(saving_criterion)
#trainer.set_stopping_criterion(stopping_criterion)

net = trainer.train(dataset, net_builder)

# net = RNN.load_model(out_dir+'/best_model.npz')
# net = trainer.resume_training(dataset, net)
