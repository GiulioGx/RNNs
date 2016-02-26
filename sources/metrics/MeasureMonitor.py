import abc

from infos import Info

__author__ = 'giulio'


class MeasureMonitor(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_symbols(self, y, t, mask)->list:
        """return a list of symbols defining the observed quantities
        given the output of the net 'y' and the target 't' """

    @abc.abstractmethod
    def update(self, measures: list):
        """updates the current measured quantity with the new estimated quantities"""

    @abc.abstractproperty
    def info(self)->Info:
        """return a 'Info' object describing the current observed values"""
