import abc

__author__ = 'giulio'


class MatrixInit(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init_matrix(self, size, dtype):
        """initialize a matrix of size='size' with dtype='dtype' """
