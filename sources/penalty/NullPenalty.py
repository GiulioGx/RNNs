import numpy
from theano import tensor as TT
from Configs import Configs
from Params import Params
from infos.Info import NullInfo
from penalty.Penalty import Penalty

__author__ = 'giulio'


class NullPenalty(Penalty):
    class Symbols(Penalty.Symbols):
        def __init__(self, W_rec):
            self.__penalty_value = TT.alloc(numpy.array(0., dtype=Configs.floatType))
            self.__penalty_grad = TT.zeros_like(W_rec, dtype=Configs.floatType)

        @property
        def penalty_grad(self):
            return self.__penalty_grad

        @property
        def penalty_value(self):
            return self.__penalty_value

        def format_infos(self, infos_symbols):
            return NullInfo(), infos_symbols

        @property
        def infos(self):
            return []

    def __init__(self):
        super().__init__()

    def compile(self, params: Params, net_symbols):
        return NullPenalty.Symbols(params.W_rec)  # FIXME