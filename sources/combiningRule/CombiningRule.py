import abc

from InfoProducer import InfoProducer

__author__ = 'giulio'


class CombiningRule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def combine(self, vector_list, n):
        """return the combination of the first 'n' vectors in vectors_list alongside some informations"""
