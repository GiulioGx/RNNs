import abc

from infos.InfoProducer import SimpleInfoProducer

__author__ = 'giulio'


class CombiningRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def combine(self, H):
        """combine (symbolically) the vectors in 'H' in some way"""
