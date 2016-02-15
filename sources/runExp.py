import sys

import theano
from SGDTrainer import SGDTrainer

from ActivationFunction import Tanh
from Configs import Configs
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.SquaredError import SquaredError
from model.RNNGrowingPolicy import RNNIncrementalGrowing
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer
from model.RNNManager import RNNManager
from output_fncs.Linear import Linear
from task.AdditionTask import AdditionTask
from task.Dataset import InfiniteDataset
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
sys.setrecursionlimit(100000)

task = AdditionTask(150, seed)
out_dir = Configs.output_dir + str(task)

# network setup
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
vars_initializer = RNNVarsInitializer(
    W_rec_init=SpectralInit(matrix_init=UniformInit(seed=seed), rho=1.2),
    W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
    W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
    b_out_init=ConstantInit(0))
net_initializer = RNNInitializer(vars_initializer, n_hidden=50)
net_growing_policy = RNNIncrementalGrowing(n_hidden_incr=5, n_hidden_max=50, n_hidden_incr_freq=1000,
                                           initializer=vars_initializer)
net_manager = RNNManager(initializer=net_initializer, activation_fnc=Tanh(), output_fnc=Linear(), growing_policy=net_growing_policy)

# setup
loss_fnc = SquaredError()

# combining_rule = OnesCombination(normalize_components=False)
combining_rule = SimplexCombination(normalize_components=True, seed=seed)
# combining_rule = SimpleSum()
dir_rule = CombinedGradients(combining_rule)
#dir_rule = LBFGSDirection(n_pairs=20)

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
# lr_rule = ConstantNormalizedStep(0.001)  # 0.01
lr_rule = GradientClipping(lr_value=0.001, clip_thr=1, normalize_wrt_dimension=False)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)

# update_rule = FixedAveraging(t=10)
update_rule = SimpleUdpate()
# update_rule = Momentum(gamma=0.1)


train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)

trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                     check_freq=200, batch_size=100, stop_error_thresh=1)

# dataset = Dataset.no_valid_dataset_from_task(size=1000, task=task)
dataset = InfiniteDataset(task=task, validation_size=10 ** 4)

net = trainer.train(dataset, net_manager)
#net = trainer.resume_training(dataset, net_manager)