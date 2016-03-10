from theano.ifelse import ifelse

from ObjectiveFunction import ObjectiveFunction
from combiningRule.CombiningRule import CombiningRule
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoList import InfoList

from infos.SymbolicInfo import SymbolicInfo

__author__ = 'giulio'


class CheckedDirection(DescentDirectionRule):

    def __init__(self, primary_strategy:DescentDirectionRule):
        self.__dir_rule = primary_strategy

    def direction(self, net, obj_fnc:ObjectiveFunction):
        gradient = obj_fnc.grad.value
        direction, dir_infos = self.__dir_rule.direction(net, obj_fnc)  # TODO add infos
        cosine = direction.cos(gradient)
        direction = net.from_tensor(ifelse(cosine >= 0, - gradient.as_tensor(), direction.as_tensor()))

        return direction, CheckedDirection.Infos(direction, cosine)

    @property
    def updates(self):
        return []

    @property
    def infos(self):
        return SimpleDescription('checked dir')  # TODO

    class Infos(SymbolicInfo):
        def __init__(self, direction, cosine):
            self.__symbols = [direction.norm(2), cosine]

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacements: list) -> Info:
            dir_norm_info = PrintableInfoElement('grad_norm', ':07.3f', symbols_replacements[0].item())
            dot_info = PrintableInfoElement('grad_dot', ':1.2f', symbols_replacements[1].item())
            info = InfoList(dir_norm_info, dot_info)
            return info
