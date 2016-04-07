import sys

import theano

from ActivationFunction import Tanh
from Configs import Configs
from Paths import Paths
from combiningRule.OnesCombination import OnesCombination
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
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer, RNNLoader
from model.RNNManager import RNNManager
from output_fncs.Logistic import Logistic
from output_fncs.Softmax import Softmax
from datasets.Dataset import InfiniteDataset
from datasets.LupusDataset import LupusDataset, PerPatienceTargets, TemporalDifferenceTargets
from datasets.TemporalOrderTask import TemporalOrderTask
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
out_dir = Configs.output_dir + 'Lupus_balanced'

# network setup
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
vars_initializer = RNNVarsInitializer(
    W_rec_init=SpectralInit(matrix_init=GaussianInit(seed=seed, std_dev=std_dev), rho=1.2),
    W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
    W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
    b_out_init=ConstantInit(0))
net_initializer = RNNInitializer(vars_initializer, n_hidden=100)
#net_initializer = RNNLoader(out_dir+'/best_model.npz')

net_growing_policy = RNNIncrementalGrowing(n_hidden_incr=5, n_hidden_max=100, n_hidden_incr_freq=1000,
                                           initializer=vars_initializer)
net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(),
                         output_fnc=Logistic())  # , growing_policy=net_growing_policy)

# setup
loss_fnc = FullCrossEntropy(single_probability_ouput=True)

#combining_rule = OnesCombination(normalize_components=True)
#combining_rule = SimplexCombination(normalize_components=True, seed=seed)
# combining_rule = SimpleSum()
#dir_rule = CombinedGradients(combining_rule)
dir_rule = Antigradient()
# dir_rule = LBFGSDirection(n_pairs=7)

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
# lr_rule = ConstantNormalizedStep(0.001)  # 0.01
lr_rule = GradientClipping(lr_value=0.001, clip_thr=1, normalize_wrt_dimension=False)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)

# update_rule = FixedAveraging(t=10)
update_rule = SimpleUdpate()
# update_rule = Momentum(gamma=0.1)


train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc, nan_check=True)

dataset = next(LupusDataset.k_fold_test_datasets(mat_file=Paths.lupus_path, k=4, strategy=TemporalDifferenceTargets()))

batch = dataset.train_set[0]

loss_monitor = LossMonitor(loss_fnc=loss_fnc)
roc_monitor = RocMonitor(score_fnc=LupusDataset.get_scores_patients)
#stopping_criterion = ThresholdCriterion(monitor=error_monitor, threshold=1. / 100)
saving_criterion = BestValueFoundCriterion(monitor=roc_monitor, mode='gt')

trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                     monitor_update_freq=50, batch_size=20)
trainer.add_monitors(dataset.train_set, 'train', loss_monitor, roc_monitor)
trainer.add_monitors(dataset.test_set, 'test', RocMonitor(score_fnc=LupusDataset.get_scores_patients))
trainer.set_saving_criterion(saving_criterion)
#trainer.set_stopping_criterion(stopping_criterion)

net = trainer.train(dataset, net_builder)

# net = RNN.load_model(out_dir+'/best_model.npz')
# net = trainer.resume_training(dataset, net)
