import abc

from infos.InfoProducer import SimpleInfoProducer
from model.Variables import Variables
from oldies.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class Penalty(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, params: Variables, net_symbols):
        """returns the compiled version"""
        
    class Symbols(SymbolicInfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def penalty_value(self):
            """returns a theano expression for penalty_value """

        @abc.abstractmethod
        def penalty_grad(self):
            """returns a theano expression for penalty_grad """


