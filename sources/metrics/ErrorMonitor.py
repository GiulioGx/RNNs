from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from metrics.RealValuedMonitor import RealValuedMonitor
from task.Dataset import Dataset


class ErrorMonitor(RealValuedMonitor):
    def __init__(self, dataset: Dataset):
        self.__dataset = dataset
        self.__error = 100
        self.__best_error = 100

    def update(self, measures: list):
        error = measures[0]

        if error < self.__best_error:
            self.__best_error = error
        self.__error = error

    def get_symbols(self, y, t) -> list:
        return [self.__dataset.computer_error(y=y, t=t)]  # XXX

    @property
    def info(self) -> Info:
        error_info = PrintableInfoElement('curr', ':.2%', self.__error)
        best_info = PrintableInfoElement('best', ':.2%', self.__best_error)
        error_group = InfoGroup('error', InfoList(error_info, best_info))
        return error_group

    @property
    def value(self):
        return self.__error
