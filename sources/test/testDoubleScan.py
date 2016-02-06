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
#net = RNN.load_model(out_dir + '/current_model.npz')

net_symbols = net.symbols

# double scan function
n_sequences = net_symbols.u.shape[2]
a_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), net.n_hidden, n_sequences)


def a_t(u_t, a_tm1, W_rec, W_in, b_rec):
    a_t = TT.dot(W_rec, TT.tanh(a_tm1)) + TT.dot(W_in, u_t) + b_rec
    da_d_w_rec = TG.jacobian(a_t.sum(axis=1), wrt=W_rec, consider_constant=[a_tm1])
    tmp=da_d_w_rec
    #tmp = da_d_w_rec.dimshuffle(1,2,0)
    tmp = tmp.reshape(shape=(W_rec.shape[0], W_rec.shape[1], a_t.shape[0], a_t.shape[1]))
    tmp = tmp.dimshuffle(2,3,0,1).sum(axis=1)
    return TT.tanh(a_t), da_d_w_rec


values, _ = T.scan(a_t, sequences=net_symbols.u,
                   outputs_info=[a_m1, None],
                   non_sequences=[net_symbols.W_rec, net_symbols.W_in, net_symbols.b_rec],
                   name='h_scan')
a = values[0]
da_d_w_rec = values[1]


def y_t(a_t, W_out, b_out):
    return TT.dot(W_out, TT.tanh(a_t)) + b_out #  XXX solo linear

values, _ = T.scan(y_t, sequences=[a],
                   outputs_info=[None],
                   non_sequences=[net_symbols.W_out, net_symbols.b_out],
                   name='y_scan')
y = values
loss = loss_fnc.value(y, net_symbols.t)

wrt = a
constants = []
vars = net_symbols.W_rec

d_C_d_wrt = T.grad(loss, wrt).sum(axis=2)
d_wrt_d_vars = TG.jacobian(wrt.sum(axis=2).flatten(), vars,
                           consider_constant=constants)

reshaped = d_wrt_d_vars.reshape(
    shape=[d_C_d_wrt.shape[0], d_C_d_wrt.shape[1], vars.shape[0], vars.shape[1]])

reshaped = da_d_w_rec


d_C_d_wrt = d_C_d_wrt.dimshuffle(0, 1, 'x')
# dot = TT.tensordot(dCdy, reshaped, axes=[[1,2], [1,2]])
# dot = TT.dot(dCdy, reshaped)
# dot = TT.alloc(0)
real_grad = TT.grad(loss, vars)


def step(A, B):
    n0 = B.shape[0]
    n1 = B.shape[1]
    n2 = B.shape[2]
    #C = B.reshape(shape=[n0, n1 * n2])
    C = B
    D = A.dot(C)
    #E = D.reshape(shape=[n1, n2])
    TT.tensordot(A, B, axes=[[0,1], [0,1]])
    return A.T.dot(B)

values, _ = T.scan(step, sequences=[d_C_d_wrt, reshaped],
                   outputs_info=[None],
                   name='prod l')

dot = values

#dot = TT.tensordot(reshaped, d_C_d_wrt, axes=[[0,1], [0,1]])
# dot = TT.alloc(0)
diff = (dot.sum(axis=0) - real_grad).norm(2)
norm_dot = dot.sum(axis=0).norm(2)
norm_real = real_grad.norm(2)
# diff = TT.alloc(0)
# exp = T.grad(loss, [net_symbols.W_rec], consider_constant=[net_symbols.h_shared])


f = theano.function([net_symbols.u, net_symbols.t], [d_C_d_wrt, reshaped, dot, real_grad, diff, norm_dot, norm_real], on_unused_input='warn')

batch = dataset.get_train_batch(1)

d_C_d_wrt, d_wrt_d_vars, dot, real_grad, diff, norm_dot, norm_real = f(batch.inputs, batch.outputs)
print('d_C_d_wrt shape', d_C_d_wrt.shape)
print('d_wrt_d_vars shape', d_wrt_d_vars.shape)
print('dot shape', dot.shape)
#print('real_grad', real_grad)
print('dc_wrt norms', li.norm(d_C_d_wrt, axis=0))

print('diff: ', diff)
print('norm dot', norm_dot)
print('norm_real', norm_real)

print('sd', d_wrt_d_vars[-1, 0, 0, :])

#print('reshaped', d_wrt_d_vars)

#print('d_C_d_wrt: ', dot[-1])
#print('real_grad: ', real_grad)