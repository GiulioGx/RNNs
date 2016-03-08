import abc
from infos.InfoProducer import SimpleInfoProducer
import theano.tensor as TT
__author__ = 'giulio'


class LossFunction(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def value(self, y, t, mask):
        """returns the loss value given the target labels 't' and the outputs 'y' and the mask 'mask'"""
