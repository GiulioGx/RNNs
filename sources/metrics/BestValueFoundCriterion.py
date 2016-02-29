from infos.InfoElement import SimpleDescription
from metrics.Criterion import Criterion
from metrics.RealValuedMonitor import RealValuedMonitor
import numpy


class BestValueFoundCriterion(Criterion):
    """this criterion is satisief if the monitored value is less(or greter if mode ="gt") to 'threshold'"""

    @property
    def infos(self):
        return SimpleDescription('decreased valued stopping criterion')

    def __init__(self, monitor: RealValuedMonitor, mode='lt'):
        self.__best_value = numpy.inf
        self.__monitor = monitor
        self.__mode = mode

        if self.__mode == 'lt':
            self.__best_value = numpy.inf
        elif self.__mode == 'gt':
            self.__best_value = - numpy.inf
        else:
            raise (ValueError('unsopported mode: {}, supported mode are "gt" and "lt""'.format(mode)))

    def is_satisfied(self) -> bool:

        if (self.__mode == 'lt' and self.__best_value > self.__monitor.value) or (
                self.__mode == 'gt' and self.__best_value < self.__monitor.value):
            self.__best_value = self.__monitor.value
            return True
        else:
            return False
