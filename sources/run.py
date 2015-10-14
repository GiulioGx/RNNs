from ActivationFunction import Tanh
from DescentDirectionRule import AntiGradientWithPenalty, MidAnglePenaltyDirection
from LearningStepRule import ConstantNormalizedStep
from RNN import RNN
from TrainingRule import TrainingRule
from tasks.AdditionTask import AdditionTask
import theano
from Configs import Configs
from Penalty import MeanPenalty, NullPenalty

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

# setup
seed = 13
task = AdditionTask(144, seed)
n_hidden = 50
activation_fnc = Tanh()
output_fnc = RNN.last_linear_fnc
loss_fnc = RNN.squared_error
penalty = MeanPenalty()
#dir_rule = AntiGradientWithPenalty(penalty, 0.001) #0.001
dir_rule = MidAnglePenaltyDirection(penalty)
lr_rule = ConstantNormalizedStep(0.005) #0.01
train_rule = TrainingRule(dir_rule, lr_rule)
net = RNN(task, activation_fnc, output_fnc, loss_fnc, n_hidden, train_rule, seed)

# train
net.train()
