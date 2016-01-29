from combiningRule.LinearCombination import LinearCombination
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SimpleInfoProducer import SimpleInfoProducer
from infos.SymbolicInfoProducer import SymbolicInfoProducer
from lossFunctions import LossFunction
from model.Variables import Variables
import theano.tensor as TT

__author__ = 'giulio'


class ObjectiveFunction(SimpleInfoProducer):
    @property
    def infos(self):
        return self.__loss_fnc.infos

    def __init__(self, loss_fnc: LossFunction):
        self.__loss_fnc = loss_fnc

    def compile(self, net, params: Variables, u, t):
        return ObjectiveFunction.Symbols(self, net, params, u, t)

    def loss(self, y, t):  # loss without penalty
        return self.__loss_fnc.value(y, t)

    @property
    def loss_fnc(self):
        return self.__loss_fnc

    class Symbols(SymbolicInfoProducer):
        def __init__(self, obj_fnc, net, params, u, t):
            self.__net = net
            self.__obj_fnc = obj_fnc

            self.__u = u
            self.__t = t
            self.__params = params

            # XXX REMOVE (?)
            self.failsafe_grad = self.__params.failsafe_grad(self.__obj_fnc.loss_fnc, self.__u, self.__t)

            self.__grad_symbols = self.__params.gradient(self.__obj_fnc.loss_fnc, self.__u, self.__t)
            self.__grad = self.__grad_symbols.value

            # y, _ = self.__net.net_output(self, u)
            # loss = loss_fnc(y, t)

            loss = self.__grad_symbols.loss_value
            self.__objective_value = loss
            self.__grad_norm = self.__grad.norm()

            # separate
            separate_info = self.__grad_symbols.infos

            # DEBUG DIFF
            #debug_diff = (self.grad - self.failsafe_grad).norm()
            debug_diff = TT.alloc(-1)

            self.__infos = separate_info + [loss, self.__grad_norm, debug_diff]

        def format_infos(self, infos_symbols):
            separate_info, infos_symbols = self.__grad_symbols.format_infos(infos_symbols)

            loss_value_info = PrintableInfoElement('value', ':07.3f', infos_symbols[0].item())
            loss_grad_info = PrintableInfoElement('grad', ':07.3f', infos_symbols[1].item())

            loss_info = InfoGroup('loss', InfoList(loss_value_info, loss_grad_info))
            obj_info = InfoGroup('obj', InfoList(loss_info, separate_info))

            norm_diff_info = PrintableInfoElement('@@', '', infos_symbols[2].item())

            info = InfoList(obj_info, norm_diff_info)

            return info, infos_symbols[loss_info.length+1:len(infos_symbols)]

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

        def grad_combination(self, strategy: LinearCombination):
            # separate time steps value
            return self.__grad_symbols.temporal_combination(strategy)
