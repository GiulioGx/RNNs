from ObjectiveFunction import ObjectiveFunction
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import SimpleDescription
from infos.InfoList import InfoList
import theano.tensor as TT


class AlternatingDirections(DescentDirectionRule):
    @property
    def infos(self):
        return InfoList(SimpleDescription('alternating_direction'), self.__main_strategy.infos)

    def __init__(self, strategy: DescentDirectionRule):
        self.__main_strategy = strategy

    @property
    def main_strategy(self):
        return self.__main_strategy

    def compile(self, net_symbols, obj_symbols):
        return AlternatingDirections.Symbols(self, net_symbols, obj_symbols)

    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols: ObjectiveFunction.Symbols):
            self.__compiled_strategy = rule.main_strategy.compute_update(net_symbols, obj_symbols)

            gradient = obj_symbols.grad

            v = TT.switch(gradient.norm() > 5, - gradient.as_tensor(),
                          self.__compiled_strategy.direction.as_tensor())  # FOXME

            self.__direction = gradient.from_tensor(v)

            self.__infos = self.__compiled_strategy.infos

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos_symbols):
            return self.__compiled_strategy.format_infos(infos_symbols)
