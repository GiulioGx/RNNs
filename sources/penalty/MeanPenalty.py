import numpy
from theano import tensor as TT

from Configs import Configs
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from model.Variables import Variables
from penalty.Penalty import Penalty
from penalty.utils import penalty_step
from theanoUtils import norm

__author__ = 'giulio'


class MeanPenalty(Penalty):
    class Symbols(Penalty.Symbols):
        def __init__(self, params, net_symbols):
            # deriv__a is a matrix n_steps * n_hidden * n_examples

            # FIXME
            W_rec = params.W_rec

            deriv_a = net_symbols.get_deriv_a(params)

            mean_deriv_a = deriv_a.mean(axis=2)

            n_steps = mean_deriv_a.shape[0]
            n_hidden = W_rec.shape[0]

            penalty_value, penalty_grad = penalty_step(mean_deriv_a,
                                                       TT.alloc(numpy.array(0., dtype=Configs.floatType)),
                                                       TT.eye(n_hidden, dtype=Configs.floatType), W_rec)

            self.__penalty_value = TT.cast(penalty_value / n_steps, dtype=Configs.floatType)
            self.__penalty_grad = TT.cast(penalty_grad / n_steps, dtype=Configs.floatType)

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

    def __init__(self):
        super().__init__()

    def compile(self, params: Variables, net_symbols):
        return MeanPenalty.Symbols(params, net_symbols)

    @property
    def infos(self):
        return SimpleDescription('mean_penalty')
