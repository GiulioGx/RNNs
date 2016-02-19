from infos.InfoElement import SimpleDescription
from metrics.Criterion import Criterion
from metrics.RealValuedMonitor import RealValuedMonitor
import numpy


class BestValueFoundCriterion(Criterion):
    """this criterion is satisief if the monitored value is less than or equal to 'threshold'"""

    @property
    def infos(self):
        return SimpleDescription('decreased valued stopping criterion')

    def __init__(self, monitor: RealValuedMonitor):
        self.__best_value = numpy.inf
        self.__monitor = monitor

    def is_satisfied(self) -> bool:
        if self.__best_value > self.__monitor.value:
            self.__best_value = self.__monitor.value
            return True
        else:
            return False
