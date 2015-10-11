from ActivationFunction import Tanh
from RNN import RNN
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

seed = 13
task = AdditionTask(144, seed)
n_hidden = 50
activation_fnc = Tanh()
output_fnc = RNN.last_linear_fnc
loss_fnc = RNN.squared_error
penalty = MeanPenalty()
net = RNN(task, activation_fnc, output_fnc, loss_fnc, n_hidden, penalty, seed)

net.train()
