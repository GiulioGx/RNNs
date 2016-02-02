import sys
import theano
from math import sqrt

from ActivationFunction import Tanh, Relu
from Configs import Configs
from SGDTrainer import SGDTrainer
from ObjectiveFunction import ObjectiveFunction
from TrainingRule import TrainingRule
from descentDirectionRule.DropoutDirection import DropoutDirection
from initialization.ConstantInit import ConstantInit
from initialization.EyeInit import EyeInit
from initialization.OrtoghonalInit import OrtoghonalInit
from initialization.SVDInit import SVDInit
from initialization.SparseGaussianInit import SparseGaussianInit
from initialization.SimplexInit import SimplexInit
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from lossFunctions.CrossEntropy import CrossEntropy
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from lossFunctions.HingeLoss import HingeLoss
from lossFunctions.NullLoss import NullLoss
from lossFunctions.SquaredError import SquaredError
from model.RNNBuilder import RNNBuilder
from model.RNNInitializer import RNNInitializer
from output_fncs.Logistic import Logistic
from output_fncs.Softmax import Softmax
from output_fncs.Linear import Linear
from task.Dataset import Dataset, InfiniteDataset
from task.MuseDataset import MuseDataset
from task.TemporalOrderTask import TemporalOrderTask
from task.XorTask import XorTask
from task.XorTaskHot import XorTaskHot
from updateRule.FixedAveragingOld import FixedAveragingOld
from updateRule.FixedAveraging import FixedAveraging
from updateRule.FixedAvgRemove import FixedAvgRemove
from updateRule.Momentum import Momentum
from updateRule.SimpleUpdate import SimpleUdpate
from combiningRule.DropoutCombination import DropoutCombination
from combiningRule.EquiangularCombination import EquiangularCombination
from combiningRule.MedianCombination import MedianCombination
from combiningRule.OnesCombination import OnesCombination
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.AlternatingDirections import AlternatingDirections
from descentDirectionRule.CombinedGradients import CombinedGradients
from descentDirectionRule.DirectionWithPenalty import DirectionWithPenalty
from initialization.GaussianInit import GaussianInit
from initialization.ZeroInit import ZeroInit
from learningRule.ArmijoStep import ArmijoStep
from learningRule.ConstantNormalizedStep import ConstantNormalizedStep
from learningRule.ConstantStep import ConstantStep
from model.RNN import RNN
from penalty.ConstantPenalty import ConstantPenalty
from penalty.MeanPenalty import MeanPenalty
from penalty.NullPenalty import NullPenalty
from task.AdditionTask import AdditionTask
from learningRule.GradientClipping import GradientClipping
from task.MultiplicationTask import MultiplicationTask

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
