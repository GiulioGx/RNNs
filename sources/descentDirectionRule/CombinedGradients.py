from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import PrintableInfoElement

__author__ = 'giulio'


class CombinedGradients(DescentDirectionRule):
    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols):
            combined_grad = obj_symbols.combined_grad
            self.__direction = combined_grad * (-1)  # FIXME - operator
            self.__infos = [self.__direction.norm()]

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos_symbols):
            info = PrintableInfoElement('dir_norm', ':07.3f', infos_symbols[0].item())
            return info, infos_symbols[info.length:len(infos_symbols)]

    def compile(self, symbol_closet, obj_symbols):
        return CombinedGradients.Symbols(self, symbol_closet, obj_symbols)
