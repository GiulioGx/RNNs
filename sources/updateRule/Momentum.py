import numpy
import theano as T

from Configs import Configs
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.SymbolicInfo import NullSymbolicInfos
from updateRule.UpdateRule import UpdateRule


class Momentum(UpdateRule):
    @property
    def updates(self) -> list:
        return self.__update_list

    def __init__(self, gamma=0.1):
        self.__gamma = gamma
        self.__update_list = []

    @property
    def gamma(self):
        return self.__gamma

    def compute_update(self, net, lr, direction):
        net_symbols = net.symbols
        self.__v = T.shared(numpy.zeros((net.n_variables, 1),
                                        dtype=Configs.floatType), broadcastable=(False, True))

        v = net.from_tensor(self.__v) * self.__gamma + (direction * lr)
        updated_params = v + net_symbols.current_params
        self.__update_list = [(self.__v, v.as_tensor())]
        return updated_params, NullSymbolicInfos()

    @property
    def infos(self):
        return InfoGroup('momentum', InfoList(PrintableInfoElement('gamma', ':2.2f', self.__gamma)))
