from InfoProducer import InfoProducer
from Infos import InfoGroup, PrintableInfoElement, InfoList, NonPrintableInfoElement
from Params import Params
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

    def compile(self, net, params: Params, u, t):
        return ObjectiveSymbols(self, net, params, u, t)

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
    def __init__(self, obj_fnc: ObjectiveFunction, net, params, u, t):
        self.__net = net
        self.__obj_fnc = obj_fnc

        # loss gradient
        y, _ = self.__net.net_output(params, u)

        loss = obj_fnc.loss_fnc(y, t)
        self.__grad = params.grad(loss)

        # add penalty
        self.__penalty_symbols = obj_fnc.penalty.compile(params, net.symbols)
        penalty_grad = self.__penalty_symbols.penalty_grad
        penalty_value = self.__penalty_symbols.penalty_value

        loss_grad_norm = self.__grad.norm()

        self.__grad.setW_rec(self.__grad.W_rec + obj_fnc.penalty_lambda * penalty_grad)  # FIXME
        self.__objective_value = loss + (penalty_value * obj_fnc.penalty_lambda)
        self.__grad_norm = self.__grad.norm()

        self.__infos = self.__penalty_symbols.infos + [loss, loss_grad_norm]


        # experimental
        y, _, W_fixes = net.experimental.net_output(params, u)
        loss = obj_fnc.loss_fnc(y, t)

        W_rec = params.W_rec  # FIXME
        self.separate_grads = T.grad(loss, W_fixes)  # FIXME

        def step(W, acc):
            return W + acc, norm(W)

        values, _ = T.scan(step, sequences=[TT.as_tensor_variable(self.separate_grads)],
                           outputs_info=[TT.zeros_like(W_rec), None],
                           non_sequences=[],
                           name='net_output',
                           mode=T.Mode(linker='cvm'),
                           n_steps=u.shape[0])

        sep_grads = values[0]
        self.__separate_grads_norms = values[1]

        self.__infos = self.__infos + [norm(self.__grad.W_rec - sep_grads[-1])] + [self.__separate_grads_norms]
        self.__gW_rec = sep_grads[-1]

    def format_infos(self, infos_symbols):
        penalty_info, infos_symbols = self.__penalty_symbols.format_infos(infos_symbols)

        loss_value_info = PrintableInfoElement('value', ':07.3f', infos_symbols[0].item())
        loss_grad_info = PrintableInfoElement('grad', ':07.3f', infos_symbols[1].item())

        loss_info = InfoGroup('loss', InfoList(loss_value_info, loss_grad_info))
        obj_info = InfoGroup('obj', InfoList(loss_info, penalty_info))
        exp_info = PrintableInfoElement('@@', '', infos_symbols[2].item())
        separate_norms_info = NonPrintableInfoElement('separate_norms', infos_symbols[3])

        info = InfoList(exp_info, obj_info, separate_norms_info)

        return info, infos_symbols[4:len(infos_symbols)]

    @property
    def infos(self):
        return self.__infos

    @property
    def objective_value(self):
        return self.__objective_value

    @property
    def grad(self):
        return self.__grad

    @property
    def grad_norm(self):
        return self.__grad_norm
