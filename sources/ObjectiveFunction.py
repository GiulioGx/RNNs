from InfoProducer import InfoProducer
from Penalty import Penalty, NullPenalty
import theano.tensor as TT
import theano as T
from theanoUtils import norm

__author__ = 'giulio'


class ObjectiveFunction(object):
    def __init__(self, loss_fnc, penalty: Penalty = NullPenalty(), penalty_lambda=0.5):
        self.__loss_fnc = loss_fnc
        self.__penalty = penalty
        self.__penalty_lambda = penalty_lambda

    def compile(self, net, W_rec, W_in, W_out, b_rec, b_out, u, t):
        return ObjectiveSymbols(self, net, W_rec, W_in, W_out, b_rec, b_out, u, t)

    def loss(self, y, t):  # loss without penalty
        return self.__loss_fnc(y, t)

    @property
    def loss_fnc(self):
        return self.__loss_fnc

    @property
    def penalty(self):
        return self.__penalty

    @property
    def penalty_lambda(self):
        return self.__penalty_lambda


class ObjectiveSymbols(InfoProducer):
    def __init__(self, obj_fnc: ObjectiveFunction, net, W_rec, W_in, W_out, b_rec, b_out, u, t):
        self.__net = net
        self.__obj_fnc = obj_fnc

        # loss gradient
        y, _ = self.__net.net_output(W_rec, W_in, W_out, b_rec, b_out, u)

        loss = obj_fnc.loss_fnc(y, t)
        self.__gW_rec, self.__gW_in, self.__gW_out, \
        self.__gb_rec, self.__gb_out = TT.grad(loss, [W_rec, W_in, W_out, b_rec, b_out])

        # add penalty
        self.__penalty_symbols = obj_fnc.penalty.compile(W_rec, W_in, W_out, b_rec, b_out, net.symbols)
        penalty_grad = self.__penalty_symbols.penalty_grad
        penalty_value = self.__penalty_symbols.penalty_value

        loss_grad_norm = norm(self.__gW_rec, self.__gW_in, self.__gW_out, self.__gb_rec, self.__gb_out)

        self.__gW_rec += obj_fnc.penalty_lambda * penalty_grad
        self.__objective_value = loss + (penalty_value * obj_fnc.penalty_lambda)
        self.__grad_norm = norm(self.__gW_rec, self.__gW_in, self.__gW_out, self.__gb_rec, self.__gb_out)

        self.__infos = self.__penalty_symbols.infos + [loss, loss_grad_norm]


        # experimental
        y, _, W_fixes = net.experimental.net_output(W_rec, W_in, W_out, b_rec, b_out, u)
        loss = obj_fnc.loss_fnc(y, t)

        # a = W_fixes[0]
        # self.__gW_rec = T.grad(loss, W_fixes)

        def step(W, acc):
            return W + acc

        values, _ = T.scan(step, sequences=[TT.as_tensor_variable(T.grad(loss, W_fixes))],
                           outputs_info=[TT.zeros_like(W_rec)],
                           non_sequences=[],
                           name='net_output',
                           mode=T.Mode(linker='cvm'),
                           n_steps=u.shape[0])

        self.__infos = self.__infos + [norm(self.__gW_rec - values[-1])]
        self.__gW_rec = values[-1]

    def format_infos(self, infos):
        desc, infos = self.__penalty_symbols.format_infos(infos)
        return '@@: {},'.format(infos[2].item()) + 'obj=[' + desc + ' loss=[value: {:07.3f}, grad: {:07.3f}]]'.format(infos[0].item(),
                                                                                  infos[1].item()), infos[
                                                                                                    3:len(infos)]

    @property
    def infos(self):
        return self.__infos

    @property
    def objective_value(self):
        return self.__objective_value

    @property
    def gW_rec(self):
        return self.__gW_rec

    @property
    def gW_in(self):
        return self.__gW_in

    @property
    def gW_out(self):
        return self.__gW_out

    @property
    def gb_rec(self):
        return self.__gb_rec

    @property
    def gb_out(self):
        return self.__gb_out

    @property
    def grad_norm(self):
        return self.__grad_norm
