import theano
from math import sqrt

from ActivationFunction import Tanh, Relu
from Configs import Configs
from NetTrainer import NetTrainer
from ObjectiveFunction import ObjectiveFunction
from TrainingRule import TrainingRule
from descentDirectionRule.DropoutDirection import DropoutDirection
from initialization.ConstantInit import ConstantInit
from initialization.EyeInit import EyeInit
from initialization.OrtoghonalInit import OrtoghonalInit
from initialization.SparseGaussianInit import SparseGaussianInit
from initialization.SimplexInit import SimplexInit
from initialization.UniformInit import UniformInit
from lossFunctions.CrossEntropy import CrossEntropy
from lossFunctions.HingeLoss import HingeLoss
from lossFunctions.NullLoss import NullLoss
from lossFunctions.SquaredError import SquaredError
from model.RnnInitializer import RnnInitializer
from output_fncs.Softmax import Softmax
from output_fncs.Linear import Linear
from task.Dataset import Dataset, InfiniteDataset
from task.TemporalOrderTask import TemporalOrderTask
from task.XorTask import XorTask
from task.XorTaskHot import XorTaskHot
from updateRule.FixedAveraging import FixedAveraging
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
from model.Rnn import Rnn
from penalty.ConstantPenalty import ConstantPenalty
from penalty.MeanPenalty import MeanPenalty
from penalty.NullPenalty import NullPenalty
from task.AdditionTask import AdditionTask
from learningRule.GradientClipping import GradientClipping
from task.MultiplicationTask import MultiplicationTask

__author__ = 'giulio'

separator = '#####################'

# setup
seed = 5656
task = XorTaskHot(70, seed)
out_dir = '/home/giulio/RNNs/models/completed/' + str(task)

net = Rnn.load_model(out_dir)

batch = task.get_batch(1)

y = net.net_ouput_numpy(batch.inputs)

print('batch_out', batch.outputs[-1, :, :])
print('y', y[0][-1])
