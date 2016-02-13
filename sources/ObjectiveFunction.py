import theano.tensor as TT
from theano.tensor.elemwise import TensorType
import theano as T

from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from infos.SymbolicInfo import SymbolicInfo
from lossFunctions import LossFunction
from model.Variables import Variables

__author__ = 'giulio'


class ObjectiveFunction(SimpleInfoProducer):
    def __init__(self, loss_fnc: LossFunction, net, params: Variables, u, t):
        self.__net = net
        self.__loss_fnc = loss_fnc

        self.__u = u
        self.__t = t
        self.__params = params

        # XXX
        self.__loss_mask = TT.tensor3(name='loss_mask')

        # XXX REMOVE (?)
        self.failsafe_grad = self.__params.failsafe_grad(self.__loss_fnc, self.__u, self.__t, self.__loss_mask)
        self.__grad = self.__params.gradient(self.__loss_fnc, self.__u, self.__t, self.__loss_mask)

        self.__objective_value = self.__grad.loss_value
        grad_norm = self.__grad.value.norm()

        # separate
        gradient_info = self.__grad.temporal_norms_infos

        # DEBUG DIFF
        # debug_diff = (self.grad - self.failsafe_grad).norm()
        debug_diff = TT.alloc(-1)

        self.__infos = ObjectiveFunction.Info(gradient_info, self.__objective_value, grad_norm, debug_diff)

    @property
    def loss_mask(self):
        return self.__loss_mask

    @property
    def infos(self):
        return self.__infos

    @property
    def grad(self):
        return self.__grad

    class Info(SymbolicInfo):
        def __init__(self, gradient_info, objective_value, grad_norm, debug_diff):
            self.__symbols = [objective_value, grad_norm, debug_diff] + gradient_info.symbols
            self.__symbolic_gradient_info = gradient_info

        def fill_symbols(self, symbols_replacements: list) -> Info:
            loss_value_info = PrintableInfoElement('value', ':07.3f', symbols_replacements[0].item())
            loss_grad_info = PrintableInfoElement('grad', ':07.3f', symbols_replacements[1].item())
            norm_diff_info = PrintableInfoElement('@@', '', symbols_replacements[2].item())

            gradient_info = self.__symbolic_gradient_info.fill_symbols(symbols_replacements[3:])

            loss_info = InfoGroup('loss', InfoList(loss_value_info, loss_grad_info))
            obj_info = InfoGroup('obj', InfoList(loss_info, gradient_info))

            info = InfoList(obj_info, norm_diff_info)
            return info

        @property
        def symbols(self):
            return self.__symbols
