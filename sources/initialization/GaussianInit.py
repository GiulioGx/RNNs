from Configs import Configs
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class GaussianInit(MatrixInit):
    def __init__(self, mean=0, std_dev=0.1, seed=Configs.seed):
        # random generator
        self.__rng = numpy.random.RandomState(seed)
        self.__mean = mean
        self.__std_dev = std_dev

    def init_matrix(self, size, dtype):
        w = numpy.asarray(self.__rng.normal(size=size, scale=self.__std_dev, loc=self.__mean), dtype=dtype)
        return w

    @property
    def infos(self):
        return InfoGroup('gaussian init strategy', InfoList(PrintableInfoElement('mean', ':2.2f', self.__mean),
                                                            PrintableInfoElement('std_dev', ':2.2f', self.__std_dev)))
