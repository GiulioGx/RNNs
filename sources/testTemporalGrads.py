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
from lossFunctions.HingeLoss import HingeLoss
from lossFunctions.NullLoss import NullLoss
from lossFunctions.SquaredError import SquaredError
from model.RNNBuilder import RNNBuilder
from model.RNNInitializer import RNNInitializer
from output_fncs.Softmax import Softmax
from output_fncs.Linear import Linear
from task.Dataset import Dataset, InfiniteDataset
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
import theano as T
import theano.tensor as TT

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

# network setup
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
rnn_initializer = RNNInitializer(W_rec_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed),
                                 W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
                                 W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
                                 b_out_init=ConstantInit(0))
net_builder = RNNBuilder(initializer=rnn_initializer, activation_fnc=Tanh(), output_fnc=Linear(), n_hidden=100)

# setup
task = AdditionTask(144, seed)
out_dir = Configs.output_dir + str(task)
loss_fnc = SquaredError()
dataset = InfiniteDataset(task=task, validation_size=10 ** 4)
net = net_builder.init_net(n_in=dataset.n_in, n_out=dataset.n_out)

V = net.symbols.compute_temporal_gradients(loss_fnc=loss_fnc)

grad_symbols = net.symbols.current_params.gradient(loss_fnc, net.symbols.u, net.symbols.t)
_, _, _, _, _, H = grad_symbols.process_temporal_components()

diff = (H.sum(axis=0) - V.sum(axis=0)).norm(2)

f = T.function([net.symbols.u, net.symbols.t], diff,
               allow_input_downcast='true',
               on_unused_input='warn',
               name='temporal_exp_step')

batch = task.get_batch(1)
diff_res = f(batch.inputs, batch.outputs)
print('batch shape ', batch.inputs.shape,  'diff ', diff_res)
