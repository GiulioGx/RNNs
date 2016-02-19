from metrics.MeasureMonitor import MeasureMonitor

import abc

__author__ = 'giulio'


class RealValuedMonitor(MeasureMonitor):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def value(self):
        """return a single real value"""
