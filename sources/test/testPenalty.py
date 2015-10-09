from RNN import RNN
from Tasks.AdditionTask import AdditionTask
import theano
from configs import Configs
from Penalty import MeanPenalty

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


batch = task.get_batch(100)

deriv_a, penalty_value, penalty_grad = net._penalty_debug(batch.inputs)
print(batch.inputs.shape)
print(deriv_a.shape)

mean_deriv_a = deriv_a.mean(axis=2)

print(deriv_a[0:10, 0:10, 0])

print(mean_deriv_a[140:145, 50:70])
print('meanMean: {}'.format(mean_deriv_a.mean().mean()))
print('grad: {}, value: {}'.format(penalty_grad, penalty_value))
