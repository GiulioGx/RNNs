import numpy

from infos.InfoElement import SimpleDescription
from initialization.GaussianInit import GaussianInit
from initialization.MatrixInit import MatrixInit


class EyeInit(MatrixInit):
    def __init__(self, scale=1, mean=0, std_dev=0.1):
        self.__scale = scale
        self.__gauss_init = GaussianInit(mean=mean, std_dev=std_dev)

    def init_matrix(self, size, dtype):
        assert (size[0] == size[1])
        e = numpy.eye(size[0], dtype=dtype) * self.__scale

        w = self.__gauss_init.init_matrix(size, dtype)

        return w + e

    @property
    def infos(self):
        return SimpleDescription('eye init strategy')
