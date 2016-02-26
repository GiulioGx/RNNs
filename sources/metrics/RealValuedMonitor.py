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

    @abc.abstractmethod
    def _update(self, new_value):
        """define the update criterion given the compute mean value"""

    def update(self, measures: list):
        acc = 0.
        for val in measures:
            acc += val[0].item()
        self._update(acc / (len(measures)))
