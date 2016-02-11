import abc
import logging

from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from model import RNN, RNNInitializer
from model.RNNInitializer import RNNVarsInitializer

__author__ = 'giulio'


class RNNGrowingPolicy(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grow(self, net: RNN):
        """expand the network according to some growing policy"""


class RNNNullGrowing(RNNGrowingPolicy):
    def grow(self, net: RNN):
        pass

    @property
    def infos(self):
        return NullInfo()


class RNNIncrementalGrowing(RNNGrowingPolicy):
    def grow(self, net: RNN):
        if (self.__counter + 1) % self.__n_hidden_incr_freq == 0 and net.n_hidden < self.__n_hidden_max:
            new_hidden_number = net.n_hidden + self.__n_hidden_incr
            net.extend_hidden_units(n_hidden=new_hidden_number, initializer=self.__initializer)
            logging.info('extending the number of hidden units to {}'.format(new_hidden_number))
        self.__counter += 1

    @property
    def infos(self):
        return InfoGroup('incremental hidden policy',
                         InfoList(PrintableInfoElement('max_units', '', self.__n_hidden_max),
                                  PrintableInfoElement('increment', '', self.__n_hidden_incr),
                                  PrintableInfoElement('frquency', '', self.__n_hidden_incr_freq)))

    def __init__(self, initializer:RNNVarsInitializer, n_hidden_max: int = 100, n_hidden_incr: int = 5, n_hidden_incr_freq: int = 2000):
        self.__n_hidden_max = n_hidden_max
        self.__n_hidden_incr = n_hidden_incr
        self.__n_hidden_incr_freq = n_hidden_incr_freq
        self.__counter = 0
        self.__initializer = initializer
