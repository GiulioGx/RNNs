from infos import Info
from infos.InfoElement import PrintableInfoElement
from lossFunctions import LossFunction
from metrics.RealValuedMonitor import RealValuedMonitor
import numpy


class LossMonitor(RealValuedMonitor):

    def __init__(self, loss_fnc:LossFunction):
        self.__loss_fnc = loss_fnc
        self.__current_value = numpy.inf

    @property
    def value(self):
        return self.__current_value

    def get_symbols(self, y, t)->list:
        return [self.__loss_fnc.value(y=y, t=t)]

    @property
    def info(self)->Info:
        return PrintableInfoElement('loss', ':07.4f', self.__current_value)

    def update(self, measures: list):
        self.__current_value = measures[0]
