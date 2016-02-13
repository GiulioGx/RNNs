import abc
from infos.InfoProducer import SimpleInfoProducer

__author__ = 'giulio'


class LossFunction(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def value(self, y, t, mask):
        """returns the loss value given the target labels 't' and the outputs 'y'.
        The mask specifies which part of the target is not used in the computation """
