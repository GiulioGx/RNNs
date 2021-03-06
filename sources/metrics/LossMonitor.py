from infos import Info
from infos.InfoElement import PrintableInfoElement
from lossFunctions import LossFunction
from metrics.RealValuedMonitor import RealValuedMonitor
import numpy


class LossMonitor(RealValuedMonitor):
    def __init__(self, loss_fnc: LossFunction):
        super().__init__(numpy.inf)
        self.__loss_fnc = loss_fnc

    def get_symbols(self, y, t, mask)->list:
        return [self.__loss_fnc.value(y=y, t=t, mask=mask)]

    @property
    def info(self)->Info:
        return PrintableInfoElement('loss', ':07.4f', self._current_value)

    def update(self, measures: list):
        self._current_value = RealValuedMonitor.mean(measures, 0)
