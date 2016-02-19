import sys

import theano

from ActivationFunction import Tanh
from Configs import Configs
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.Antigradient import Antigradient
from descentDirectionRule.CombinedGradients import CombinedGradients
from descentDirectionRule.LBFGSUpdate import LBFGSDirection
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

seed = 13
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
    W_rec_init=SpectralInit(matrix_init=GaussianInit(seed=seed, std_dev=std_dev), rho=1.7),
    W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
    W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
    b_out_init=ConstantInit(0))
net_initializer = RNNInitializer(vars_initializer, n_hidden=50)
net_growing_policy = RNNIncrementalGrowing(n_hidden_incr=5, n_hidden_max=50, n_hidden_incr_freq=1000,
                                           initializer=vars_initializer)
net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(), output_fnc=Softmax())#, growing_policy=net_growing_policy)

# setup
task = TemporalOrderTask(100, seed)
out_dir = Configs.output_dir + str(task)+'new'
loss_fnc = FullCrossEntropy(single_probability_ouput=False)

# # HF init
# bias_value = 0.5
# n_conns = 25
# std_dev = sqrt(0.12)
# init_strategies = {'W_rec': RandomConnectionsInit(n_connections_per_unit=n_conns, std_dev=std_dev, columnwise=False),
#                    'W_in': RandomConnectionsInit(n_connections_per_unit=n_conns, std_dev=0.1, columnwise=True),
#                    'W_out': RandomConnectionsInit(n_connections_per_unit=n_conns, std_dev=std_dev, columnwise=False),
#                    'b_rec': ConstantInit(bias_value), 'b_out': ConstantInit(bias_value)}

# penalty strategy
# penalty = MeanPenalty()
# penalty = ConstantPenalty(c=5)
# penalty = MeanPenalty()

# direction strategy
# dir_rule = AntiGradient()
# dir_rule = AntiGradientWithPenalty(penalty, 1) #0.001
# dir_rule = MidAnglePenaltyDirection(penalty)
# dir_rule = FrozenGradient(penalty)
# dir_rule = SepareteGradient()

# combining_rule = OnesCombination(normalize_components=False)
combining_rule = SimplexCombination(normalize_components=True, seed=seed)
# combining_rule = SimpleSum()
dir_rule = CombinedGradients(combining_rule)
#dir_rule = Antigradient()
#dir_rule = LBFGSDirection(n_pairs=7)

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
# lr_rule = ConstantNormalizedStep(0.001)  # 0.01
lr_rule = GradientClipping(lr_value=0.005, clip_thr=1, normalize_wrt_dimension=False)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)

# update_rule = FixedAveraging(t=10)
update_rule = SimpleUdpate()
# update_rule = Momentum(gamma=0.1)


train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)

trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                     check_freq=200, batch_size=100, stop_error_thresh=1)

# dataset = Dataset.no_valid_dataset_from_task(size=1000, task=task)
dataset = InfiniteDataset(task=task, validation_size=10 ** 4, n_batches=5)

net = trainer.train(dataset, net_builder)

# net = RNN.load_model(out_dir+'/best_model.npz')
# net = trainer.resume_training(dataset, net)
