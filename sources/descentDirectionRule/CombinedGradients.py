from combiningRule.CombiningRule import CombiningRule
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import NonPrintableInfoElement, PrintableInfoElement
from infos.InfoList import InfoList
from theanoUtils import norm

__author__ = 'giulio'


class CombinedGradients(DescentDirectionRule):
    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols):

            combined_grad, norms = obj_symbols.combined_grad(rule.strategy)
            self.__direction = combined_grad * (-1)  # FIXME - operator
            self.__infos = [norms, self.__direction.norm()]

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos_symbols):
            separate_norms_info = NonPrintableInfoElement('separate_norms', infos_symbols[0])
            dir_norm = PrintableInfoElement('dir_norm', ':07.3f', infos_symbols[1].item())
            info = InfoList(separate_norms_info, dir_norm)
            return info, infos_symbols[info.length:len(infos_symbols)]

    def __init__(self, combining_strategy: CombiningRule):
        self.__strategy = combining_strategy

    def compile(self, symbol_closet, obj_symbols):
        return CombinedGradients.Symbols(self, symbol_closet, obj_symbols)

    @property
    def strategy(self):
        return self.__strategy
