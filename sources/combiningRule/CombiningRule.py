import abc

from infos.SimpleInfoProducer import SimpleInfoProducer
from infos.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class CombiningRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, vector_list, n):
        """combine the vectors in 'vector_list' in some way"""

    class Symbols(SymbolicInfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractproperty
        def combination(self):
            """return a symbol for the computed combination"""
