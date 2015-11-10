import abc

from InfoProducer import InfoProducer

__author__ = 'giulio'


class CombiningRule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, vector_list, n):
        """combine the vectors in 'vector_list' in some way"""

    class Symbols(InfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractproperty
        def combination(self):
            """return a symbol for the computed combination"""
