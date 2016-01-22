import abc

from infos.SimpleInfoProducer import SimpleInfoProducer
from infos.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class CombiningRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, H):
        """combine the vectors in 'H' in some way"""

    class Symbols(SymbolicInfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractproperty
        def combination(self):
            """return a symbol for the computed combination"""
