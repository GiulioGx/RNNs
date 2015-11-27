from Configs import Configs
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from learningRule.LearningRule import LearningStepRule
from updateRule.UpdateRule import UpdateRule
import theano as T
import numpy


class Momentum(UpdateRule):
    def __init__(self, gamma=0.3):
        self.__gamma = gamma

    @property
    def gamma(self):
        return self.__gamma

    def compile(self, net, net_symbols, lr_symbols: LearningStepRule.Symbols,
                dir_symbols: DescentDirectionRule.Symbols):
        return Momentum.Symbols(self, net, net_symbols, lr_symbols, dir_symbols)

    @property
    def infos(self):
        return InfoGroup('momentum', InfoList(PrintableInfoElement('gamma', ':2.2f', self.__gamma)))

    class Symbols(UpdateRule.Symbols):
        def __init__(self, rule, net, net_symbols, lr_symbols: LearningStepRule.Symbols,
                     dir_symbols: DescentDirectionRule.Symbols):
            self.__v = T.shared(numpy.zeros((net.n_variables, 1),
                                            dtype=Configs.floatType), broadcastable=(False, True))

            v = net.from_tensor(self.__v) * rule.gamma + (dir_symbols.direction * lr_symbols.learning_rate)
            updated_params = v + net_symbols.current_params
            self.__update_list = [(self.__v, v.as_tensor())] + net_symbols.current_params.update_list(updated_params)

        @property
        def infos(self):
            return []

        @property
        def update_list(self):
            return self.__update_list

        def format_infos(self, infos):
            return NullInfo(), infos
