from RNN import RNN
from Tasks.AdditionTask import AdditionTask
import theano
from configs import Configs
from penalties import MeanPenalty

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
n_hidden = 100
activation_fnc = RNN.relu
output_fnc = RNN.last_linear_fnc
loss_fnc = RNN.squared_error
penalty = MeanPenalty()
net = RNN(task, activation_fnc, output_fnc, loss_fnc, n_hidden, penalty, seed)

net.train()
