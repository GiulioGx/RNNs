import abc
from infos.InfoProducer import SimpleInfoProducer
import theano.tensor as TT
__author__ = 'giulio'


class LossFunction(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # The mask specifies which part of the target is not used in the computation
        self.__mask = TT.tensor3(name='mask')

    @property
    def mask(self):
        return self.__mask

    @abc.abstractmethod
    def value(self, y, t):
        """returns the loss value given the target labels 't' and the outputs 'y' """
