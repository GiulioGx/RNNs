from ObjectiveFunction import ObjectiveFunction
from combiningRule.CombiningRule import CombiningRule
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoList import InfoList

__author__ = 'giulio'


class CombinedGradients(DescentDirectionRule):

    @property
    def infos(self):
        return InfoList(SimpleDescription('combined_gradients'), self.__combining_strategy.infos)

    def __init__(self, strategy: CombiningRule):
        self.__combining_strategy = strategy

    @property
    def combining_strategy(self):
        return self.__combining_strategy

    def compile(self, net_symbols, obj_symbols):
        return CombinedGradients.Symbols(self, net_symbols, obj_symbols)

    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols: ObjectiveFunction.Symbols):
            self.__combined_grad_symbols = obj_symbols.grad_combination(rule.combining_strategy)
            self.__direction = -self.__combined_grad_symbols.value
            grad_dot = self.__direction.dot(obj_symbols.grad)

            diff_norm = (-obj_symbols.grad - self.__direction).norm()

            self.__infos = self.__combined_grad_symbols.infos + [self.__direction.norm(), grad_dot / (
                self.__direction.norm() * obj_symbols.grad.norm()),
                                                                 diff_norm]

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos_symbols):
            combining_info, infos_symbols = self.__combined_grad_symbols.format_infos(infos_symbols)
            dir_norm_info = PrintableInfoElement('dir_norm', ':07.3f', infos_symbols[0].item())
            dot_info = PrintableInfoElement('grad_dot', ':1.2f', infos_symbols[1].item())
            norm_diff_info = PrintableInfoElement('@@', '', infos_symbols[2].item())
            info = InfoList(combining_info, dir_norm_info, dot_info, norm_diff_info)

            return info, infos_symbols[info.length:len(infos_symbols)]
