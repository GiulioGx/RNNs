from Configs import Configs
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class OrtoghonalInit(MatrixInit):
    def __init__(self, mean=0, std_var=0.1, scale = 1, seed=Configs.seed):
        # random generator
        self.__rng = numpy.random.RandomState(seed)
        self.__mean = mean
        self.__std_var = std_var
        self.__scale = scale

    def init_matrix(self, size, dtype):
        # TODO assert size
        values = self.__rng.uniform(size=size)
        u, _, _ = numpy.linalg.svd(values)
        u = u * self.__scale
        return u.astype(dtype)
