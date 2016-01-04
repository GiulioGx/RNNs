import abc

from infos.InfoElement import SimpleDescription
from infos.SimpleInfoProducer import SimpleInfoProducer

__author__ = 'giulio'


class MatrixInit(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init_matrix(self, size, dtype):
        """initialize a matrix of size='size' with dtype='dtype' """

    @property
    def infos(self):
        return SimpleDescription('No description available')
