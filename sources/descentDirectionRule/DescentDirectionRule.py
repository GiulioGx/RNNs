import abc

from InfoProducer import InfoProducer

__author__ = 'giulio'


class DescentDirectionRule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, symbol_closet, obj_symbols):
        """returns the compiled version"""
        
    class Symbols(InfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def direction(self):
            """return a symbol for the computed descent direction"""

