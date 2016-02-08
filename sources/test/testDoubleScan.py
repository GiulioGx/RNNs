import numpy
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
from model.RNNBuilder import RNNBuilder
from model.RNNInitializer import RNNInitializer
from output_fncs.Linear import Linear
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
rnn_initializer = RNNInitializer(W_rec_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed),
                                 W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
                                 W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
                                 b_out_init=ConstantInit(0))
net_builder = RNNBuilder(initializer=rnn_initializer, activation_fnc=Tanh(), output_fnc=Linear(), n_hidden=100)

# setup
task = XorTaskHot(144, seed)
out_dir = Configs.output_dir + str(task)
loss_fnc = SquaredError()

combining_rule = OnesCombination(normalize_components=False)
dir_rule = CombinedGradients(combining_rule)

dataset = InfiniteDataset(task=task, validation_size=10 ** 4)
net = net_builder.init_net(n_in=dataset.n_in, n_out=dataset.n_out)
# net = RNN.load_model(out_dir + '/current_model.npz')

net_symbols = net.symbols

# DOUBLE scan function
n_sequences = net_symbols.u.shape[2]
a_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), net.n_hidden, n_sequences)


def a_t(u_t, a_tm1, W_rec, W_in, b_rec):
    a_t = TT.dot(W_rec, TT.tanh(a_tm1)) + TT.dot(W_in, u_t) + b_rec
    return a_t


values, _ = T.scan(a_t, sequences=net_symbols.u,
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
loss = loss_fnc.value(y, net_symbols.t)

wrt = a
vars = net_symbols.W_rec

# Trick to get dC/da[k]
scan_node = a.owner.inputs[0].owner
assert isinstance(scan_node.op, theano.scan_module.scan_op.Scan)
n_pos = scan_node.op.n_seqs + 1
init_a = scan_node.inputs[n_pos]
d_C_d_wrt = T.grad(loss, init_a)
d_C_d_wrt = d_C_d_wrt[1:]


real_grad = TT.grad(loss, vars)

A = d_C_d_wrt.dimshuffle(0, 2, 1)


def smart_step(A, u_t, a_tm1):

    a_t = TT.dot(net_symbols.W_rec, TT.tanh(a_tm1)) + TT.dot(net_symbols.W_in, u_t) + net_symbols.b_rec

    vars = net_symbols.W_rec
    n1 = vars.shape[0]
    n2 = vars.shape[1]
    N = a_t.shape[1]
    n_hid = a_t.shape[0]

    B = TG.jacobian(a_t.flatten(), wrt=vars, consider_constant=[a_tm1])
    B = B.reshape(shape=(n_hid, N, n1, n2))
    B = B.dimshuffle(1,0,2,3)

    A = A.reshape(shape=(1, N * n_hid))
    B = B.reshape(shape=(N*n_hid, n1*n2))

    C = A.dot(B)
    C = C.reshape(shape=(n1, n2))
    return a_t, C

values, _ = T.scan(smart_step, sequences=[A, net_symbols.u],
                   outputs_info=[a_m1, None],
                   name='outer_scan',
                   n_steps=A.shape[0])

dot = values[1]

#dot = TT.zeros(shape=(a.shape[0], real_grad.shape[0], real_grad.shape[1]))

#dot = TT.tensordot(A,B, axes=[[1,2], [1, 2]])

diff = (dot.sum(axis=0) - real_grad).norm(2)
norm_dot = dot.sum(axis=0).norm(2)
norm_real = real_grad.norm(2)

h = TT.tanh(a).sum(axis=2)
#
# f = theano.function([net_symbols.u, net_symbols.t],
#                     [A, B, dot, real_grad, diff, norm_dot, norm_real, h],
#                     on_unused_input='warn')

# f = theano.function([net_symbols.u, net_symbols.t],
#                     [A, B, dot, real_grad, diff, norm_dot, norm_real, h],
#                     on_unused_input='warn')
# print('l: ', batch.inputs.shape[0])
# print('d_C_d_wrt shape', d_C_d_wrt.shape)
# print('d_wrt_d_vars shape', d_wrt_d_vars_numpy.shape)
# print('dot shape', dot.shape)
#
# print('diff: ', diff)
# print('norm dot', norm_dot)
# print('norm_real', norm_real)
#
# sum = 0
# for l in range(d_C_d_wrt.shape[0]):
#     for e in range(d_C_d_wrt.shape[1]):
#         for n1 in range(d_C_d_wrt.shape[2]):
#             sum += d_C_d_wrt[l, e, n1] * d_wrt_d_vars_numpy[l, e, n1, 0, 0]
#
# print('sum', sum)
# print(dot.sum(axis=0)[0, 0])
# print(real_grad[0, 0])

f = theano.function([net_symbols.u, net_symbols.t],
                    [diff],
                    on_unused_input='warn')
batch = dataset.get_train_batch(batch_size=100)
diff = f(batch.inputs, batch.outputs)
#print(diff[0].shape)
print('diff: ', diff)

