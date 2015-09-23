from RNN import RNN
from Tasks.AdditionTask import AdditionTask
import theano

__author__ = 'giulio'

separator = '#####################'


# ###THEANO CONFIG ### #
floatX = theano.config.floatX
device = theano.config.device
print(separator)
print('THEANO CONFIG')
print('device: ' + device)
print('floatX: ' + floatX)
print(separator)

seed = 13
task = AdditionTask(13, seed)
n_hidden = 30
activation_fnc = RNN.sigmoid
output_fnc = RNN.last_linear_fnc
net = RNN(task, activation_fnc, output_fnc, n_hidden, seed)

sequence = task.get_batch(5).inputs
output_sequence = net.net_output(sequence)
print(output_sequence)
