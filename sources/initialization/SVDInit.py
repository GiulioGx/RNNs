from Configs import Configs
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.GaussianInit import GaussianInit
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class SVDInit(MatrixInit):
    def __init__(self, matrix_init: MatrixInit, rho: float, seed: int = Configs.seed):
        # random generator
        self.__matrix_init = matrix_init
        self.__rho = rho
        self.__rng = numpy.random.RandomState(seed)

    def init_matrix(self, size, dtype):
        w = self.__matrix_init.init_matrix(size, dtype)
        assert len(w.shape) == 2
        assert w.shape[0] == w.shape[1]

        u, s, v = numpy.linalg.svd(w)

        s_normalized = numpy.ones_like(s)

        self.__mean = 0
        self.__std_dev = 5

        s_normalized = numpy.asarray(self.__rng.normal(size=s.shape, scale=self.__std_dev, loc=self.__mean),
                                     dtype=dtype)
        s_normalized = numpy.asarray(self.__rng.uniform(size=s.shape, low=-1, high=1),
                                     dtype=dtype) * 1.2
        #s_normalized *= self.__rho / max(abs(s_normalized))

        # s_normalized = self.__matrix_init.init_matrix(size=(1, s.shape[1]), dtype=dtype)
        w_normalized = u.dot(numpy.diag(s_normalized)).dot(v)

        return w_normalized

    @property
    def infos(self):
        return InfoGroup('SVD init meta strategy', InfoList(PrintableInfoElement('rho', ':2.2f', self.__rho),
                                                            InfoGroup('strategy',
                                                                      InfoList(self.__matrix_init.infos))))


if __name__ == '__main__':
    n = 5
    size = (n, n)
    rho = 1.2
    print('Generating a matrix with size: {} with the spectral initialization strategy (radius: {:2.2f})'.format(size,
                                                                                                                 rho))
    strategy = GaussianInit(mean=0, std_dev=0.3, seed=45435435)
    strategy = SVDInit(matrix_init=strategy, rho=1.2)
    w = strategy.init_matrix(size, dtype='float32')
    print('generated matrix is\n', w)
    print('spectral radius of the generated matrix is {:2.2f}'.format(MatrixInit.spectral_radius(w)))
    _, s, _ = numpy.linalg.svd(w)
    print('singular values:\n', s)
    print('Done...')
