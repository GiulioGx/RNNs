import theano.tensor as TT
import theano as T
import numpy

from Configs import Configs
from model import Variables


class FixedAveraging(object):
    def __init__(self, t=7):
        self.__t = t

    @property
    def t(self):
        return self.__t

    def compile(self, net):
        return FixedAveraging.Symbols(self, net)

    class Symbols(object):
        def __init__(self, strategy, net):
            self.__counter = TT.cast(T.shared(0, name='average_t'), dtype='int32')
            self.__acc = T.shared(numpy.zeros((net.n_variables, 1), dtype=Configs.floatType))  # FIXME current params
            self.__strategy = strategy

        def apply_average(self, params: Variables):
            vec = params.as_tensor()
            new_counter, new_acc, new_params_vec = TT.switch(self.__counter+1 >= self.__strategy.t,
                                                             [0, vec, (self.__acc + vec) / self.__strategy.t],
                                                             [self.__counter + 1, self.__acc + vec, vec])
            update_dictionay = [(self.__counter, new_counter), (self.__acc, new_acc)]

            new_params = params.net.from_tensor(new_params_vec)
            return new_params, update_dictionay
