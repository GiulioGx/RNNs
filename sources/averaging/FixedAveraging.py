import theano.tensor as TT
import theano as T
from Configs import Configs
from averaging.AveragingRule import AveragingRule
from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement
from model import Variables


class FixedAveraging(AveragingRule):
    def infos(self):
        return PrintableInfoElement('t', ':02d', self.__t)

    def __init__(self, t=7):
        self.__t = t

    @property
    def t(self):
        return self.__t

    def compile(self, net):
        return FixedAveraging.Symbols(self, net)

    class Symbols(AveragingRule.Symbols):

        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos

        def __init__(self, strategy, net):
            self.__counter = T.shared(0, name='avg_t')
            #self.__acc = T.shared(numpy.zeros((net.n_variables, 1), dtype=Configs.floatType), broadcastable=(False, False))
            self.__acc = T.shared(net.symbols.get_numeric_vector, name='avg_acc', broadcastable=(False, False))
            self.__strategy = strategy

        def apply_average(self, params: Variables):
            vec = params.as_tensor()

            condition = self.__counter + 1 >= self.__strategy.t-1
            new_counter = TT.switch(condition, 0, self.__counter + 1)

            mean_point = (self.__acc + vec)/TT.cast(self.__strategy.t, dtype=Configs.floatType)
            new_acc = TT.switch(condition, mean_point, self.__acc + vec)
            new_params_vec = TT.switch(condition, mean_point, vec)

            # new_counter, new_acc, new_params_vec = TT.switch(self.__counter+1 >= self.__strategy.t,
            #                                                 [0, vec, (self.__acc + vec) / self.__strategy.t],
            #                                                 [self.__counter + 1, self.__acc + vec, vec])

            update_dictionay = [(self.__counter, new_counter), (self.__acc, new_acc)]

            new_params = params.net.from_tensor(new_params_vec)
            return new_params, update_dictionay
