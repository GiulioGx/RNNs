import theano
import theano as T

from ActivationFunction import Tanh
from Configs import Configs
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from lossFunctions.SquaredError import SquaredError
from model.RNNManager import RNNManager
from model.RNNInitializer import RNNInitializer
from output_fncs.Linear import Linear
from task.AdditionTask import AdditionTask
from task.Dataset import InfiniteDataset

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

# network setup
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
rnn_initializer = RNNInitializer(W_rec_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed),
                                 W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
                                 W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
                                 b_out_init=ConstantInit(0))
net_builder = RNNManager(initializer=rnn_initializer, activation_fnc=Tanh(), output_fnc=Linear(), n_hidden=100)

# setup
task = AdditionTask(144, seed)
out_dir = Configs.output_dir + str(task)
loss_fnc = SquaredError()
dataset = InfiniteDataset(task=task, validation_size=10 ** 4)
net = net_builder.get_net(n_in=dataset.n_in, n_out=dataset.n_out)

V = net.symbols.compute_temporal_gradients(loss_fnc=loss_fnc)

grad_symbols = net.symbols.current_params.gradient(loss_fnc, net.symbols.u, net.symbols.t)
_, _, _, _, _, H = grad_symbols.process_temporal_components()

diff = (H.sum(axis=0) - V.sum(axis=0)).norm(2)

f = T.function([net.symbols.u, net.symbols.t], diff,
               allow_input_downcast='true',
               on_unused_input='warn',
               name='temporal_exp_step')

batch = task.get_batch(1)
diff_res = f(batch.inputs, batch.outputs)
print('batch shape ', batch.inputs.shape,  'diff ', diff_res)
