from metrics.MeasureMonitor import MeasureMonitor

import abc

__author__ = 'giulio'


class RealValuedMonitor(MeasureMonitor):
    __metaclass__ = abc.ABCMeta

    def __init__(self, init_value: float):
        self._current_value = init_value

    @property
    def value(self):
        return self._current_value

    @staticmethod
    def mean(measures: list, index:int=0):
        acc = 0.
        for val in measures:
            acc += val[index].item()
        return acc / len(measures)
