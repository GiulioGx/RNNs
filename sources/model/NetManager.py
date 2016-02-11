import abc

from infos.InfoProducer import SimpleInfoProducer

__author__ = 'giulio'


class NetManager(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_net(self, n_in: int, n_out: int):
        """initialize the network according to some scheme for input sequences composed of 'n_in' elements
        and output sequences composed of 'n_out' elements for each time step"""

    @abc.abstractmethod
    def grow_net(self):
        """expand the network according to some growing policy"""
