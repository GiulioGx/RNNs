from Configs import Configs
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class SpectralInit(MatrixInit):
    def __init__(self, matrix_init: MatrixInit, rho: float):
        # random generator
        self.__matrix_init = matrix_init
        self.__rho = rho

    def init_matrix(self, size, dtype):
        w = self.__matrix_init.init_matrix(size, dtype)
        assert len(w.shape) == 2
        assert w.shape[0] == w.shape[1]

        rho = MatrixInit.spectral_radius(w)
        return w / rho * self.__rho

    @property
    def infos(self):
        return InfoGroup('spectral init meta strategy', InfoList(PrintableInfoElement('rho', ':2.2f', self.__rho),
                                                                 InfoGroup('strategy',
                                                                           InfoList(self.__matrix_init.infos))))
