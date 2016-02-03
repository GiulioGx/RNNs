from ObjectiveFunction import ObjectiveFunction
from combiningRule.CombiningRule import CombiningRule
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoList import InfoList
import theano.tensor as TT

from infos.SymbolicInfo import SymbolicInfo

__author__ = 'giulio'


class CombinedGradients(DescentDirectionRule):
    def __init__(self, strategy: CombiningRule):
        self.__combining_strategy = strategy

    def direction(self, net_symbols, obj_fnc:ObjectiveFunction):
        gradients_combination, combining_strategy_symbolic_info = obj_fnc.grad.temporal_combination(
            self.__combining_strategy)
        direction = - gradients_combination
        grad_dot = direction.cos(obj_fnc.grad.value)

        return direction, CombinedGradients.Infos(combining_strategy_symbolic_info, direction.norm(), grad_dot)

    @property
    def infos(self):
        return InfoList(SimpleDescription('combined_gradients'), self.__combining_strategy.infos)

    class Infos(SymbolicInfo):
        def __init__(self, combining_strategy_info, dir_norm, grad_dot):
            self.__combining_strategy_info = combining_strategy_info
            self.__symbols = [dir_norm, grad_dot] + combining_strategy_info.symbols

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacements: list) -> Info:
            dir_norm_info = PrintableInfoElement('dir_norm', ':07.3f', symbols_replacements[0].item())
            dot_info = PrintableInfoElement('grad_dot', ':1.2f', symbols_replacements[1].item())
            combining_info = self.__combining_strategy_info.fill_symbols(symbols_replacements[2:])
            info = InfoList(combining_info, dir_norm_info, dot_info)
            return info
