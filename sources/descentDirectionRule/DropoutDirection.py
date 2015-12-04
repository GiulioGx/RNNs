from theano.tensor.shared_randomstreams import RandomStreams

from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList


class DropoutDirection(DescentDirectionRule):
    """a wrapper of DescentDirectionRule which randomly 'turn off' some components"""

    def __init__(self, dir_rule: DescentDirectionRule, drop_rate, seed: int = Configs.seed):
        self.__dir_rule = dir_rule
        self.__drop_rate = drop_rate
        self.__srng = RandomStreams(seed=seed)

    def generate_coeff(self, size):
        return self.__srng.choice(size=size, a=[0, 1], replace=True, p=[self.drop_rate, 1 - self.__drop_rate],
                                  dtype=Configs.floatType)

    @property
    def infos(self):
        return InfoGroup('dropout_direction',
                         InfoList(PrintableInfoElement('drop_rate', ':2.2f', self.__drop_rate), self.__dir_rule.infos))

    @property
    def drop_rate(self):
        return self.__drop_rate

    @property
    def dir_rule(self):
        return self.__dir_rule

    def compile(self, net_symbols, obj_symbols: ObjectiveFunction.Symbols):
        return DropoutDirection.Symbols(self, net_symbols, obj_symbols)

    class Symbols(DescentDirectionRule.Symbols):
        def __init__(self, rule, net_symbols, obj_symbols: ObjectiveFunction.Symbols):
            self.__dir_symbols = rule.dir_rule.compile(net_symbols, obj_symbols)

            v = self.__dir_symbols.direction.as_tensor()
            coeff = rule.generate_coeff(v.shape)

            self.__direction = self.__dir_symbols.direction.net.from_tensor(v * coeff)

            self.__infos = self.__dir_symbols.infos

        @property
        def direction(self):
            return self.__direction

        @property
        def infos(self):
            return self.__infos

        def format_infos(self, infos):
            return self.__dir_symbols.format_infos(infos)
