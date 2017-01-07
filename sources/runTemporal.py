import sys

import theano, numpy

from ActivationFunction import Tanh, Relu
from Configs import Configs
from combiningRule.ClippingCombination import ClippingCombination
from combiningRule.OnesCombination import OnesCombination
from combiningRule.ReducedSimplexCombination import ReducedSimplexCombination
from combiningRule.SimplexCombination import SimplexCombination
from combiningRule.TemporalClippingCombination import TemporalClippingCombination
from combiningRule.TimeSmoothingCombination import TimeSmoothingCombination
from combiningRule.UniformRandomCombination import UniformRandomCombination
from descentDirectionRule.Antigradient import Antigradient
from descentDirectionRule.CheckedDirection import CheckedDirection
from descentDirectionRule.CombinedGradients import CombinedGradients
from descentDirectionRule.DropoutDirection import DropoutDirection
from descentDirectionRule.LBFGSDirection import LBFGSDirection
from initialization.ConstantInit import ConstantInit
from initialization.EigenInit import EigenInit
from initialization.GaussianInit import GaussianInit
from initialization.SVDInit import SVDInit
from initialization.SparseGaussianInit import SparseGaussianInit
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from learningRule.AdaptiveStep import AdaptiveStep
from learningRule.GradientClipping import GradientClipping
from learningRule.ProbabilisticSearch import ProbabilisticSearch
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from metrics.BestValueFoundCriterion import BestValueFoundCriterion
from metrics.ErrorMonitor import ErrorMonitor
from metrics.LossMonitor import LossMonitor
from metrics.ThresholdCriterion import ThresholdCriterion
from model.RNNGrowingPolicy import RNNIncrementalGrowing
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer, RNNLoader
from model.RNNManager import RNNManager
from output_fncs.Softmax import Softmax
from datasets.Dataset import InfiniteDataset
from datasets.TemporalOrderTask import TemporalOrderTask
from training.SGDTrainer import SGDTrainer
from training.TrainingRule import TrainingRule
from updateRule.CyclicUpdate import CyclicUdpate
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

seed = 14
Configs.seed = seed

task = TemporalOrderTask(200, seed)
out_dir = Configs.output_dir + str(task) + '_' + str(seed)
# network setup
std_dev = 0.05  # 0.14 Tanh # 0.21 Relu
mean = 0
vars_initializer = RNNVarsInitializer(
    # W_rec_init=SpectralInit(matrix_init=GaussianInit(mean=mean, std_dev=.01, seed=seed), rho=1.2),
    W_rec_init=EigenInit(mean=1.2, std_dev=0.01, seed=seed),
    W_in_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed),
    W_out_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed), b_rec_init=ConstantInit(0),
    b_out_init=ConstantInit(0))
# vars_initializer = RNNVarsInitializer(
#     W_rec_init=GaussianInit(seed=seed, std_dev=std_dev),
#     W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
#     W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
#     b_out_init=ConstantInit(0))
net_initializer = RNNInitializer(vars_initializer, n_hidden=50)
net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(),
                         output_fnc=Softmax())  # , growing_policy=net_growing_policy)

# setup

loss_fnc = FullCrossEntropy(single_probability_ouput=False)

# combining_rule = UniformRandomCombination(normalize_components=True, seed=seed)
# combining_rule = OnesCombination(normalize_components=True)
# combining_rule = SimplexCombination(normalize_components=True, seed=seed)
# combining_rule = TimeSmoothingCombination(normalize_components=False, seed=seed)
# combining_rule = ReducedSimplexCombination(keep_ratio=0.5, normalize_components=True, seed=seed)

# combining_rule = TemporalClippingCombination(thr=0.1, clip_style="nothing_for_now")

# combining_rule = SimpleSum()
# dir_rule = CombinedGradients(combining_rule)
# dir_rule = CheckedDirection(dir_rule, max_cos=0, max_dir_norm=numpy.inf)
dir_rule = Antigradient()
# dir_rule = LBFGSDirection(n_pairs=7)
# dir_rule = DropoutDirection(dir_rule=dir_rule, drop_rate=0.7, seed=seed)

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
# lr_rule = ConstantNormalizedStep(0.001)  # 0.01
lr_rule = GradientClipping(lr_value=0.005, clip_thr=0.1, clip_style='l1')  # 0.01
# lr_rule = AdaptiveStep(init_lr=0.001, num_tokens=50, prob_augment=0.4, sliding_window_size=50, steps_int_the_past=5,
#                               beta_augment=1.1, beta_lessen=0.1, seed=seed)
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)

# update_rule = FixedAveraging(t=10)
update_rule = SimpleUdpate()
# update_rule = Momentum(gamma=0.1)
# update_rule = CyclicUdpate()


train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)

# dataset = Dataset.no_valid_dataset_from_task(size=1000, datasets=datasets)
dataset = InfiniteDataset(task=task, validation_size=10 ** 4, n_batches=5)
val_batches = dataset.validation_set
print("validation set batches:")
for b in val_batches:
    print("\t", b.inputs.shape)

loss_monitor = LossMonitor(loss_fnc=loss_fnc)
error_monitor = ErrorMonitor(dataset=dataset, error_fnc=task.error_fnc)
stopping_criterion = ThresholdCriterion(monitor=error_monitor, threshold=1. / 100)
saving_criterion = BestValueFoundCriterion(monitor=error_monitor)

trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                     monitor_update_freq=200, batch_size=20) # update_freq 200
trainer.add_monitors(dataset.validation_set, "validation", loss_monitor, error_monitor)
trainer.set_saving_criterion(saving_criterion)
trainer.set_stopping_criterion(stopping_criterion)

net = trainer.train(dataset, net_builder)

# net = RNN.load_model(out_dir+'/best_model.npz')
# net = trainer.resume_training(dataset, net_builder)
