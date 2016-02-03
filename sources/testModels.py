import theano as T

from model.RNN import RNN
from task.AdditionTask import AdditionTask

__author__ = 'giulio'

separator = '#####################'

# setup
seed = 2
task = AdditionTask(144, seed)
out_dir = '/home/giulio/snoopy2models/' + str(task)+'/model.npz'
task = AdditionTask(144, seed)

net = RNN.load_model(out_dir)

batch = task.get_batch(5)

y = net.net_ouput_numpy(batch.inputs)

print('shape', len(y[-1]))


print('batch_out', batch.outputs[-1, :, :])
print('y', y[0][-1])


error = task.error_fnc(net.symbols.t, net.symbols.y_shared)
error_fnc = T.function([net.symbols.u, net.symbols.t], [error], name='error_fnc')

print('error:', error_fnc(batch.inputs, batch.outputs))

