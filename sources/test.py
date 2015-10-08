from RNN import RNN
from Tasks.AdditionTask import AdditionTask
import theano
from configs import Configs

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
activation_fnc = RNN.tanh
output_fnc = RNN.last_linear_fnc
loss_fnc = RNN.squared_error
net = RNN(task, activation_fnc, output_fnc, loss_fnc, n_hidden, seed)

batch = task.get_batch(2)
sequence = batch.inputs
targets = batch.outputs
print('input sequence shape = {}'.format(sequence.shape))
output_sequence = net.net_output(sequence)
print(str(batch))
print('output sequence shape = {}'.format(output_sequence.shape))
print(output_sequence)
output_sequence[-1, :, 0] = 0.69337243
output_sequence[-1, :, 1] = 0.60164523
loss_error = loss_fnc(output_sequence, targets)
print('init loss error = {}'.format(loss_error))
loss_error = loss_fnc(targets, targets)
print('target loss error = {}'.format(loss_error))
error = task.error_fnc(targets, targets)
print('target error = {}'.format(error))


net.train()
