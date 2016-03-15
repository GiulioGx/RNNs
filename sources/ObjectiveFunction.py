import theano.tensor as TT
from theano.tensor.elemwise import TensorType
import theano as T

from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from infos.SymbolicInfo import SymbolicInfo
from lossFunctions.LossFunction import LossFunction
from model.Variables import Variables

__author__ = 'giulio'


class ObjectiveFunction(SimpleInfoProducer):  # XXX is this class needed?
    def __init__(self, loss_fnc: LossFunction, net, params: Variables, u, t, mask):
        self.__net = net
        self.__loss_fnc = loss_fnc

        self.__u = u
        self.__t = t
        self.__params = params

        # XXX

        # XXX REMOVE (?)
        self.failsafe_grad, _ = self.__net.symbols.failsafe_grad(u=u, t=t, mask=mask, params=self.__params,
                                                                 obj_fnc=self)
        self.__grad, self.__objective_value = self.__net.symbols.gradient(u=u, t=t, mask=mask, params=self.__params,
                                                                          obj_fnc=self)

        grad_norm = self.__grad.value.norm()

        # separate
        gradient_info = self.__grad.temporal_norms_infos

        # DEBUG DIFF
        # debug_diff = (self.grad.value - self.failsafe_grad).norm()
        debug_diff = TT.alloc(-1)

        self.__infos = ObjectiveFunction.Info(gradient_info, self.__objective_value, grad_norm, debug_diff,
                                              net.symbols.mask)

    @property
    def current_loss(self):
        return self.__objective_value

    def value(self, y, t, mask):
        return self.__loss_fnc.value(y=y, t=t, mask=mask)

    @property
    def loss_mask(self):
        return self.__loss_fnc.mask

    @property
    def infos(self):
        return self.__infos

    @property
    def grad(self):
        return self.__grad

    class Info(SymbolicInfo):
        def __init__(self, gradient_info, objective_value, grad_norm, debug_diff, mask):
            # n_selected_temporal_losses = TT.switch(mask.sum(axis=1) > 0, 1, 0).sum(axis=1).sum()
            n_selected_temporal_losses = LossFunction.num_examples_insting_temp_loss(mask)

            self.__symbols = [objective_value, grad_norm, debug_diff,
                              n_selected_temporal_losses] + gradient_info.symbols
            self.__symbolic_gradient_info = gradient_info

        def fill_symbols(self, symbols_replacements: list) -> Info:
            loss_value_info = PrintableInfoElement('value', ':07.3f', symbols_replacements[0].item())
            loss_grad_info = PrintableInfoElement('grad', ':07.3f', symbols_replacements[1].item())
            norm_diff_info = PrintableInfoElement('@@', '', symbols_replacements[2].item())
            n_loss_info = PrintableInfoElement('##n', '', symbols_replacements[3])

            gradient_info = self.__symbolic_gradient_info.fill_symbols(symbols_replacements[4:])

            loss_info = InfoGroup('loss', InfoList(loss_value_info, loss_grad_info))
            obj_info = InfoGroup('obj', InfoList(loss_info, gradient_info))

            info = InfoList(obj_info, norm_diff_info, n_loss_info)
            return info

        @property
        def symbols(self):
            return self.__symbols
