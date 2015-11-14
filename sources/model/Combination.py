import abc

from infos.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class Combination(SymbolicInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def value(self):
        """return a 'Variables' class which represents the combination"""
