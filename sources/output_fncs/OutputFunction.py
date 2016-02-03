import abc
from infos.InfoProducer import SimpleInfoProducer

__author__ = 'giulio'


class OutputFunction(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def value(self, x):
        """returns an output value for the given 'x'"""
