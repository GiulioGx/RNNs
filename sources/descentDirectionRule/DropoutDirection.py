from theano.tensor.shared_randomstreams import RandomStreams

from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo

__author__ = 'giulio'


class DropoutDirection(DescentDirectionRule):
    """a wrapper of DescentDirectionRule which randomly 'turn off' some components"""

    def __init__(self, dir_rule: DescentDirectionRule, drop_rate, seed: int = Configs.seed):
        self.__dir_rule = dir_rule
        self.__drop_rate = drop_rate
        self.__srng = RandomStreams(seed=seed)

    @property
    def updates(self) -> list:
        return []

    def direction(self, net, obj_fnc: ObjectiveFunction):
        v, dir_info = self.__dir_rule.direction(net, obj_fnc)
        v = v.as_tensor()
        coeff = self.__generate_coeff(v.shape)
        return net.from_tensor(v * coeff), DropoutDirection.Infos(dir_info, v)

    def __generate_coeff(self, size):
        return self.__srng.choice(size=size, a=[0, 1], replace=True, p=[self.__drop_rate, 1 - self.__drop_rate],
                                  dtype=Configs.floatType)

    @property
    def infos(self):
        return InfoGroup('dropout_direction',
                         InfoList(PrintableInfoElement('drop_rate', ':2.2f', self.__drop_rate), self.__dir_rule.infos))

    class Infos(SymbolicInfo):  # TODO
        def __init__(self, dir_strategy_info, v):
            self.__combining_strategy_info = dir_strategy_info
            self.__symbols = dir_strategy_info.symbols

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacements: list) -> Info:
            combining_info = self.__combining_strategy_info.fill_symbols(symbols_replacements[0:])
            info = InfoList(combining_info)
            return info
