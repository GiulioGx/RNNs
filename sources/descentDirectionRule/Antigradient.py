from ObjectiveFunction import ObjectiveFunction
from combiningRule.CombiningRule import CombiningRule
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoList import InfoList
import theano.tensor as TT

from infos.SymbolicInfo import SymbolicInfo

__author__ = 'giulio'


class Antigradient(DescentDirectionRule):

    def __init__(self, strategy: CombiningRule):
        self.__combining_strategy = strategy

    def direction(self, net, obj_fnc:ObjectiveFunction):
        gradient = obj_fnc.grad.value
        direction = - gradient

        return direction, Antigradient.Infos(direction)

    @property
    def updates(self):
        return []

    @property
    def infos(self):
        return SimpleDescription('antigradient')

    class Infos(SymbolicInfo):
        def __init__(self, direction):
            self.__symbols = [direction.norm(2)]

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacements: list) -> Info:
            dir_norm_info = PrintableInfoElement('grad_norm', ':07.3f', symbols_replacements[0].item())
            info = InfoList(dir_norm_info)
            return info
