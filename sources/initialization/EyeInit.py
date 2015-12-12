import numpy

from initialization.GaussianInit import GaussianInit
from initialization.MatrixInit import MatrixInit


class EyeInit(MatrixInit):
    def __init__(self, scale=1, mean = 0, std_dev=0.1):
        self.__scale = scale
        self.__gauss_init = GaussianInit(mean=mean, std_var=std_dev)

    def init_matrix(self, size, dtype):
        e = numpy.eye(size[0], dtype=dtype) * self.__scale  # FIXME assert

        w = self.__gauss_init.init_matrix(size, dtype)

        return w + e
