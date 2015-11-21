import abc

from infos.SimpleInfoProducer import SimpleInfoProducer
from infos.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class DescentDirectionRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, net_symbols, obj_symbols):
        """returns the compiled version"""
        
    class Symbols(SymbolicInfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractproperty
        def direction(self):
            """return a symbol for the computed descent direction"""

