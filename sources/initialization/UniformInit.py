from Configs import Configs
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class UniformInit(MatrixInit):
    def __init__(self, high=1, low=-1, seed=Configs.seed):
        # random generator
        self.__rng = numpy.random.RandomState(seed)
        self.__high = high
        self.__low = low

    def init_matrix(self, size, dtype):
        w = numpy.asarray(self.__rng.uniform(size=size, low=self.__low, high=self.__high), dtype=dtype)
        return w

    @property
    def infos(self):
        return InfoGroup('uniform init strategy', InfoList(PrintableInfoElement('low', ':2.2f', self.__low),
                                                           PrintableInfoElement('high', ':2.2f', self.__high)))
