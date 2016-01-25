import theano.tensor as TT
import theano as T
from Configs import Configs
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from learningRule.LearningRule import LearningStepRule
from updateRule.UpdateRule import UpdateRule
from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement


class FixedAveragingOld(UpdateRule):
    @property
    def infos(self):
        return InfoGroup("average_old", InfoList(PrintableInfoElement('t', ':02d', self.__t)))

    def __init__(self, t=7):
        self.__t = t

    @property
    def t(self):
        return self.__t

    def compile(self, net, net_symbols, lr_symbols: LearningStepRule.Symbols,
                dir_symbols: DescentDirectionRule.Symbols):
        return FixedAveragingOld.Symbols(self, net, net_symbols, lr_symbols, dir_symbols)

    class Symbols(UpdateRule.Symbols):
        @property
        def infos(self):
            return []

        def format_infos(self, infos):
            return NullInfo(), infos

        def __init__(self, strategy, net, net_symbols, lr_symbols: LearningStepRule.Symbols,
                     dir_symbols: DescentDirectionRule.Symbols):
            updated_params = net_symbols.current_params + (dir_symbols.direction * lr_symbols.learning_rate)
            self.__counter = T.shared(0, name='avg_counter')
            self.__acc = T.shared(net.symbols.get_numeric_vector, name='avg_acc', broadcastable=(False, False))
            self.__strategy = strategy

            vec = updated_params.as_tensor()

            condition = self.__counter + 1 >= self.__strategy.t - 1
            new_counter = TT.switch(condition, 0, self.__counter + 1)

            mean_point = (self.__acc + vec) / TT.cast(self.__strategy.t, dtype=Configs.floatType)
            new_acc = TT.switch(condition, mean_point, self.__acc + vec)
            new_params_vec = TT.switch(condition, mean_point, vec)

            self.__update_list = [(self.__counter, new_counter),
                                  (self.__acc, new_acc)] + net_symbols.current_params.update_list(
                net.from_tensor(new_params_vec))
            # self.__avg_params = updated_params.net.from_tensor(new_params_vec)

        @property
        def update_list(self):
            return self.__update_list
