import abc

from infos.InfoProducer import SimpleInfoProducer

__author__ = 'giulio'


class Rule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def updates(self)->list:
        """returns list of updates for theano"""
