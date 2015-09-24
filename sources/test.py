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
print('floatType: ' + floatX)
print(separator)

seed = 13
task = AdditionTask(13, seed)
n_hidden = 30
activation_fnc = RNN.sigmoid
output_fnc = RNN.last_linear_fnc
loss_fnc = RNN.squared_error
net = RNN(task, activation_fnc, output_fnc, loss_fnc, n_hidden, seed)

batch = task.get_batch(5)
sequence = batch.inputs
targets = batch.outputs
print('input sequence shape = {}'.format(sequence.shape))
output_sequence = net.net_output(sequence)
print('output sequence shape = {}'.format(output_sequence.shape))
print(output_sequence)
loss_error = loss_fnc(output_sequence, targets)
print('loss error = {}'.format(loss_error))


net.train()