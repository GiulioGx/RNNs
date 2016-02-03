import abc

from ObjectiveFunction import ObjectiveFunction
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoProducer import SimpleInfoProducer
from infos.SymbolicInfo import SymbolicInfo

__author__ = 'giulio'


class LearningStepRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_lr(self, net, obj_fnc: ObjectiveFunction, direction):
        """return the computed lr"""

    class Infos(SymbolicInfo):
        def __init__(self, lr_value):
            self.__symbols = [lr_value]

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacedments: list) -> Info:
            return PrintableInfoElement('lr', ':02.2e', symbols_replacedments[0].item())
