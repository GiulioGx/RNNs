from Configs import Configs
from infos.InfoElement import SimpleDescription
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class GaussianInit(MatrixInit):

    def __init__(self, mean=0, std_var=0.1, seed=Configs.seed):
        # random generator
        self.__rng = numpy.random.RandomState(seed)
        self.__mean = mean
        self.__std_var = std_var

    def init_matrix(self, size, dtype):

        w = numpy.asarray(self.__rng.normal(size=size, scale=self.__std_var, loc=self.__mean), dtype=dtype)
        return w

    @property
    def infos(self):
        return SimpleDescription('gaussian init strategy')

