from Configs import Configs
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.GaussianInit import GaussianInit
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class SpectralInit(MatrixInit):
    """A decorator for MatrixInit. It scale the matrix so to have a given spectral radius"""
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

if __name__ == '__main__':
    n = 5
    size = (n, n)
    rho=1.2
    print('Generating a matrix with size: {} with the spectral initialization strategy (radius: {:2.2f})'.format(size, rho))
    strategy = GaussianInit(mean=0, std_dev=0.3)
    strategy = SpectralInit(matrix_init=strategy, rho=1.2)
    w = strategy.init_matrix(size, dtype='float32')
    print('generated matrix is\n', w)
    print('spectral radius of the generated matrix is {:2.2f}'.format(MatrixInit.spectral_radius(w)))
    print('Done...')
