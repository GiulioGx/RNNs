import theano

from ActivationFunction import Tanh
from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from combiningRule.OnesCombination import OnesCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from lossFunctions.SquaredError import SquaredError
from model.RNN import RNN
from model.RNNManager import RNNManager
from model.RNNInitializer import RNNInitializer
from output_fncs.Linear import Linear
from datasets.AdditionTask import AdditionTask
from datasets.Dataset import InfiniteDataset
import theano as T
import theano.gradient as TG
import theano.tensor as TT

import numpy.linalg as li
from datasets.XorTaskHot import XorTaskHot

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
task = XorTaskHot(144, seed)
out_dir = Configs.output_dir + str(task)
loss_fnc = SquaredError()

combining_rule = OnesCombination(normalize_components=False)
dir_rule = CombinedGradients(combining_rule)

dataset = InfiniteDataset(task=task, validation_size=10 ** 4)
net = net_builder.get_net(n_in=dataset.n_in, n_out=dataset.n_out)
net = RNN.load_model(out_dir + '/current_model.npz')

net_symbols = net.symbols

# Trick to get dC/dh[k]
scan_node = net_symbols.h_shared.owner.inputs[0].owner
assert isinstance(scan_node.op, T.scan_module.scan_op.Scan)
n_pos = scan_node.op.n_seqs + 1
init_h = scan_node.inputs[n_pos]
print('npos ', n_pos)
print('init_h ', init_h)
print('node', scan_node)


loss = loss_fnc.value(net_symbols.y_shared, net_symbols.t)

wrt = net_symbols.y_shared
constants = []
vars = net_symbols.W_rec

d_C_d_wrt = T.grad(loss, wrt).sum(axis=2)
d_wrt_d_vars = TG.jacobian(wrt.sum(axis=2).flatten(), vars,
                           consider_constant=constants)

reshaped = d_wrt_d_vars.reshape(
    shape=[d_C_d_wrt.shape[0], d_C_d_wrt.shape[1], vars.shape[0], vars.shape[1]])

d_C_d_wrt = d_C_d_wrt.dimshuffle(0, 1, 'x')
# dot = TT.tensordot(dCdy, reshaped, axes=[[1,2], [1,2]])
# dot = TT.dot(dCdy, reshaped)
# dot = TT.alloc(0)
real_grad = TT.grad(loss, vars)


def step(A, B):
    n0 = B.shape[0]
    n1 = B.shape[1]
    n2 = B.shape[2]
    C = B.reshape(shape=[n0, n1 * n2])
    D = A.T.dot(C)
    E = D.reshape(shape=[n1, n2])
    return E


values, _ = T.scan(step, sequences=[d_C_d_wrt, reshaped],
                   name='net_output_scan')

dot = values
# dot = TT.alloc(0)
diff = (dot[-1] - real_grad).norm(2)
# diff = TT.alloc(0)
# exp = T.grad(loss, [net_symbols.W_rec], consider_constant=[net_symbols.h_shared])

###
# Trick to get dC/dh[k]
scan_node = net_symbols.h_shared.owner.inputs[0].owner
assert isinstance(scan_node.op, theano.scan_module.scan_op.Scan)
n_pos = scan_node.op.n_seqs + 1
print('npos', n_pos)
init_h = scan_node.inputs[n_pos]
dcdh = T.grad(loss, init_h).sum(axis=2)
dcdh = dcdh[1:]
###

f = theano.function([net_symbols.u, net_symbols.t], [d_C_d_wrt, reshaped, dot, diff, dcdh], on_unused_input='warn')

batch = dataset.get_train_batch(1)

d_C_d_wrt, d_wrt_d_vars, dot, diff, dcdh_numpy = f(batch.inputs, batch.outputs)
print('dCdy shape', d_C_d_wrt.shape)
print('dydw_out shape', d_wrt_d_vars.shape)
print('dot shape', dot.shape)
print('diff', diff)

print('others', sum(sum(sum(dot[0:-2]))))
print(dot[145])

norms = li.norm(dcdh_numpy, axis=1)
print('dcdh_numpy norms', norms)
