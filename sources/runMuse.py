import sys

import theano

from ActivationFunction import Tanh
from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from SGDTrainer import SGDTrainer
from TrainingRule import TrainingRule
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from model.RNNBuilder import RNNBuilder
from model.RNNInitializer import RNNInitializer
from output_fncs.Logistic import Logistic
from task.MuseDataset import MuseDataset
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
sys.setrecursionlimit(1000000)

# network setup
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
rnn_initializer = RNNInitializer(W_rec_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed),
                                 W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
                                 W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
                                 b_out_init=ConstantInit(0))
net_builder = RNNBuilder(initializer=rnn_initializer, activation_fnc=Tanh(), output_fnc=Logistic(), n_hidden=100)

# setup
out_dir = Configs.output_dir + 'MuseDataset'
loss_fnc = FullCrossEntropy()

# combining_rule = OnesCombination(normalize_components=True)
combining_rule = SimplexCombination(normalize_components=True, seed=seed)
# combining_rule = SimpleSum()
dir_rule = CombinedGradients(combining_rule)

# lr_rule = ConstantNormalizedStep(0.001)  # 0.01
lr_rule = GradientClipping(lr_value=0.03, clip_thr=0.1)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)
obj_fnc = ObjectiveFunction(loss_fnc)

# update_rule = FixedAveraging(t=10)
update_rule = SimpleUdpate()
# update_rule = Momentum(gamma=0.1)

train_rule = TrainingRule(dir_rule, lr_rule, update_rule)

trainer = SGDTrainer(train_rule, obj_fnc, output_dir=out_dir, max_it=10 ** 10,
                     check_freq=200, batch_size=100, stop_error_thresh=0.1)

# dataset = Dataset.no_valid_dataset_from_task(size=1000, task=task)
dataset = MuseDataset(seed=seed, pickle_file_path='/home/giulio/RNNs/datasets/polyphonic/musedata/MuseData.pickle')

net = trainer.train(dataset, net_builder, seed=seed)

# net = RNN.load_model(out_dir+'/current_model.npz')
# net = trainer.resume_training(dataset, net)
