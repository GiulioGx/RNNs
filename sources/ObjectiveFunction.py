import theano.tensor as TT
import theano as T

from InfoProducer import InfoProducer
from combiningRule.SImpleSum import SimpleSum
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoElement import NonPrintableInfoElement, PrintableInfoElement
from Params import Params
from penalty.Penalty import Penalty
from penalty.NullPenalty import NullPenalty
from theanoUtils import norm, cos_between_dirs

__author__ = 'giulio'


class ObjectiveFunction(object):
    def __init__(self, loss_fnc, penalty: Penalty = NullPenalty(), penalty_lambda=0.5):
        self.__loss_fnc = loss_fnc
        self.__penalty = penalty
        self.__penalty_lambda = penalty_lambda

    def compile(self, net, params: Params, u, t):
        return ObjectiveFunction.Symbols(self, net, params, u, t)

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

    class Symbols(InfoProducer):
        def __init__(self, obj_fnc, net, params, u, t):
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

            # separate time steps grad
            combined_grad, separate_norms = params.grad_combining_steps(obj_fnc.loss_fnc, SimpleSum(), u, t)

            self.__infos = self.__infos + [combined_grad.norm()-self.__grad_norm] + [separate_norms]

        def format_infos(self, infos_symbols):
            penalty_info, infos_symbols = self.__penalty_symbols.format_infos(infos_symbols)

            loss_value_info = PrintableInfoElement('value', ':07.3f', infos_symbols[0].item())
            loss_grad_info = PrintableInfoElement('grad', ':07.3f', infos_symbols[1].item())

            loss_info = InfoGroup('loss', InfoList(loss_value_info, loss_grad_info))
            obj_info = InfoGroup('obj', InfoList(loss_info, penalty_info))
            exp_info = PrintableInfoElement('@@', '', infos_symbols[2].item())
            separate_norms_info = NonPrintableInfoElement('separate_norms', infos_symbols[3])

            info = InfoList(exp_info, obj_info, separate_norms_info)

            return info, infos_symbols[info.length:len(infos_symbols)]

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