from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from metrics.RealValuedMonitor import RealValuedMonitor
from task.Dataset import Dataset


class ErrorMonitor(RealValuedMonitor):

    def __init__(self, dataset: Dataset, error_fnc):
        super().__init__(100)
        self.__dataset = dataset
        self.__best_error = 100
        self.__error_fnc = error_fnc

    def update(self, measures: list):

        new_value = RealValuedMonitor.mean(measures, 0)
        if new_value < self.__best_error:
            self.__best_error = new_value
        self._current_value = new_value

    def get_symbols(self, y, t, mask) -> list:
        return [self.__error_fnc(y=y, t=t, mask=mask)]  # XXX

    @property
    def info(self) -> Info:
        error_info = PrintableInfoElement('curr', ':.2%', self._current_value)
        best_info = PrintableInfoElement('best', ':.2%', self.__best_error)
        error_group = InfoGroup('error', InfoList(error_info, best_info))
        return error_group
