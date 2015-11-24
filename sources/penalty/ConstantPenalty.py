from theano import tensor as TT

from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from model.Variables import Variables
from penalty.Penalty import Penalty
from penalty.utils import deriv_a_T_wrt_a1
from theanoUtils import norm

__author__ = 'giulio'


class ConstantPenalty(Penalty):

    def __init__(self, c=1):
        self.__c = c

    @property
    def c(self):
        return self.__c

    class Symbols(Penalty.Symbols):
        def __init__(self, penalty, params, net_symbols):
            # deriv__a is a matrix n_steps * n_hidden * n_examples

            deriv_a = net_symbols.get_deriv_a(params)

            mean_deriv_a = deriv_a.mean(axis=2)
            #n_steps = TT.cast(mean_deriv_a.shape[0], dtype=Configs.floatType)

            W_rec = params.W_rec # FIXME

            A = deriv_a_T_wrt_a1(W_rec, mean_deriv_a)

            self.__penalty_value = ((A ** 2).sum() - penalty.c) ** 2
            self.__penalty_grad = TT.grad(self.__penalty_value, [W_rec], consider_constant=[deriv_a])[0]

            self.__infos = [self.__penalty_value, norm(self.__penalty_grad)]

        @property
        def penalty_grad(self):
            return self.__penalty_grad

        @property
        def penalty_value(self):
            return self.__penalty_value

        def format_infos(self, info_symbols):
            penalty_value_info = PrintableInfoElement('value', ':07.3f', info_symbols[0].item())
            penalty_grad_info = PrintableInfoElement('grad', ':07.3f', info_symbols[1].item())
            info = InfoGroup('penalty', InfoList(penalty_value_info, penalty_grad_info))
            return info, info_symbols[info.length:len(info_symbols)]

        @property
        def infos(self):
            return self.__infos

    def compile(self, params: Variables, net_symbols):
        return ConstantPenalty.Symbols(self, params, net_symbols)

    @property
    def infos(self):
        return InfoGroup('constant_penalty', InfoList(PrintableInfoElement('c', ':2.2f', self.__c)))


