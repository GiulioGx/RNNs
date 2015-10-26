import abc

from InfoProducer import InfoProducer
from Params import Params

__author__ = 'giulio'


class Penalty(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, params: Params, net_symbols):
        """returns the compiled version"""
        
    class Symbols(InfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def penalty_value(self):
            """returns a theano expression for penalty_value """

        @abc.abstractmethod
        def penalty_grad(self):
            """returns a theano expression for penalty_grad """


