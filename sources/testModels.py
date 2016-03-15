import theano as T

from model.RNN import RNN
from datasets.AdditionTask import AdditionTask
from datasets.TemporalOrderTask import TemporalOrderTask

__author__ = 'giulio'

separator = '#####################'

# setup
seed = 22
out_dir = '/home/giulio/temporal_runs/144/non_incr_100/snoopy4/temporal_order, min_length: 144'+'/best_model.npz'
task = TemporalOrderTask(144, seed)

net = RNN.load_model(out_dir)

batch = task.get_batch(10000)

y = net.net_ouput_numpy(batch.inputs)

print('shape', len(y[-1]))
print('batch_out', batch.outputs[-1, :, :])
print('y', y[0][-1])


error = task.error_fnc(net.symbols.t, net.symbols.y_shared)
error_fnc = T.function([net.symbols.u, net.symbols.t], [error*100], name='error_fnc')

print('error:', error_fnc(batch.inputs, batch.outputs))

