from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import PrintableInfoElement
from infos.InfoList import InfoList

__author__ = 'giulio'


class CombinedGradients(DescentDirectionRule):
    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols):
            combined_grad = obj_symbols.combined_grad
            self.__direction = combined_grad * (-1)  # FIXME - operator
            grad_dot = self.__direction.dot(obj_symbols.grad)
            self.__infos = [self.__direction.norm(), grad_dot/(self.__direction.norm()*obj_symbols.grad.norm())]

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos_symbols):
            dir_norm_info = PrintableInfoElement('dir_norm', ':07.3f', infos_symbols[0].item())
            dot_info = PrintableInfoElement('grad_dot', ':1.2f', infos_symbols[1].item())

            info = InfoList(dir_norm_info, dot_info)

            return info, infos_symbols[info.length:len(infos_symbols)]

    def compile(self, symbol_closet, obj_symbols):
        return CombinedGradients.Symbols(self, symbol_closet, obj_symbols)
