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

    def compile(self, net, update_params: Variables):
        return FixedAveraging.Symbols(self, net, update_params)

    class Symbols(AveragingRule.Symbols):
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos

        def __init__(self, strategy, net, update_paramas: Variables):
            self.__counter = T.shared(0, name='avg_counter')
            # self.__acc = T.shared(numpy.zeros((net.n_variables, 1),
            #                                  dtype=Configs.floatType), broadcastable=(False, False))
            self.__acc = T.shared(net.symbols.get_numeric_vector, name='avg_acc', broadcastable=(False, False))
            self.__strategy = strategy

            vec = update_paramas.as_tensor()

            condition = self.__counter + 1 >= self.__strategy.t - 1
            new_counter = TT.switch(condition, 0, self.__counter + 1)

            mean_point = (self.__acc + vec) / TT.cast(self.__strategy.t, dtype=Configs.floatType)
            new_acc = TT.switch(condition, mean_point, self.__acc + vec)
            new_params_vec = TT.switch(condition, mean_point, vec)

            self.__update_list = [(self.__counter, new_counter), (self.__acc, new_acc)]
            self.__avg_params = update_paramas.net.from_tensor(new_params_vec)

        @property
        def averaged_params(self):
            return self.__avg_params

        @property
        def update_list(self):
            return self.__update_list
