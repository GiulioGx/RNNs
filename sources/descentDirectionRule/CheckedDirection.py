from theano.ifelse import ifelse

from ObjectiveFunction import ObjectiveFunction
from combiningRule.CombiningRule import CombiningRule
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList

from infos.SymbolicInfo import SymbolicInfo
from theanoUtils import ifelse_vars

__author__ = 'giulio'


class CheckedDirection(DescentDirectionRule):
    def __init__(self, primary_strategy: DescentDirectionRule, max_cos: float = 0, max_dir_norm: float = 0.5):
        self.__dir_rule = primary_strategy
        self.__max_cos = max_cos
        self.__max_dir_norm = max_dir_norm

    def direction(self, net, obj_fnc: ObjectiveFunction):
        gradient = obj_fnc.grad.value
        direction, dir_infos = self.__dir_rule.direction(net, obj_fnc)  # TODO add infos
        cosine = direction.cos(gradient)
        direction = ifelse_vars(cosine >= self.__max_cos, - gradient, direction, net=net)
        direction = ifelse_vars(direction.norm(2) > self.__max_dir_norm, -gradient, direction, net=net)

        return direction, CheckedDirection.Infos(direction, cosine)

    @property
    def updates(self):
        return []

    @property
    def infos(self):
        return InfoGroup('checked_dir', InfoList(InfoGroup('primary_strategy', InfoList(self.__dir_rule.infos)),
                                                 PrintableInfoElement('max_cos', ':.2f', self.__max_cos),
                                                 PrintableInfoElement('max_dir_norm', ':.2f', self.__max_dir_norm)))

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
