import numpy
import theano
import time

from ActivationFunction import Tanh
from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from combiningRule.OnesCombination import OnesCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.RNNVarsInitializer import RNNVarsInitializer
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from lossFunctions.CrossEntropy import CrossEntropy
from lossFunctions.SquaredError import SquaredError
from model.RNN import RNN
from model.RNNManager import RNNManager
from model.RNNInitializer import RNNInitializer
from output_fncs.Linear import Linear
from output_fncs.Softmax import Softmax
from task.AdditionTask import AdditionTask
from task.Dataset import InfiniteDataset
import theano as T
import theano.gradient as TG
import theano.tensor as TT
import numpy.linalg as li
from task.XorTaskHot import XorTaskHot

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
vars_initializer = RNNVarsInitializer(
    W_rec_init=SpectralInit(matrix_init=UniformInit(seed=seed), rho=1.2),
    W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
    W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
    b_out_init=ConstantInit(0))
net_initializer = RNNInitializer(vars_initializer, n_hidden=100)
net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(), output_fnc=Softmax())

# setup
task = XorTaskHot(144, seed)
out_dir = Configs.output_dir + str(task)
loss_fnc = CrossEntropy()

combining_rule = OnesCombination(normalize_components=False)
dir_rule = CombinedGradients(combining_rule)

dataset = InfiniteDataset(task=task, validation_size=10 ** 4)
net = net_builder.get_net(n_in=dataset.n_in, n_out=dataset.n_out)
# net = RNN.load_model(out_dir + '/current_model.npz')

net_symbols = net.symbols

# DOUBLE scan function
n_sequences = net_symbols.u.shape[2]
a_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), net.n_hidden, n_sequences)
loss_mask = TT.tensor3(name='mask')

def fill(W):
    return W

values, updates_scan = T.scan(fill, non_sequences=[net_symbols.W_rec], outputs_info=[None],
                              name='replicate_scan', n_steps=net_symbols.u.shape[0])

W3 = values


def a_t(u_t, a_tm1, W_rec, W_in, b_rec):
    a_t = TT.dot(W_rec, TT.tanh(a_tm1)) + TT.dot(W_in, u_t) + b_rec
    return a_t


values, updates_scan = T.scan(a_t, sequences=[net_symbols.u],
                   outputs_info=[a_m1],
                   non_sequences=[net_symbols.W_rec, net_symbols.W_in, net_symbols.b_rec],
                   name='a_scan')
a = values


def y_t(a_t, W_out, b_out):  # FOXME vectorial output function
    return TT.dot(W_out, TT.tanh(a_t)) + b_out  # XXX solo linear


values, _ = T.scan(y_t, sequences=[a],
                   outputs_info=[None],
                   non_sequences=[net_symbols.W_out, net_symbols.b_out],
                   name='y_scan')
y = values
loss = loss_fnc.value(y, net_symbols.t, loss_mask)


#real_grad = TT.grad(loss, net_symbols.W_rec)
#norm_real = real_grad.norm(2)

grad = TT.grad(loss, net_symbols.W_rec)

h = TT.tanh(a).sum(axis=2)

f = theano.function([net_symbols.u, net_symbols.t, loss_mask],
                    [grad],
                    on_unused_input='warn', updates=updates_scan)
print('getting batch...')
batch = dataset.get_train_batch(batch_size=2)
print('computing gradients...')
t1 = time.time()
result_numpy = f(batch.inputs, batch.outputs, batch.mask)
t2 = time.time()
print('elapsed_time: {}s'.format(t2 - t1))
print('diff: ', result_numpy[0].shape)
print('sum: ', result_numpy[0][6])