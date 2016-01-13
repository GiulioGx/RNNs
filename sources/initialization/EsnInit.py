import numpy
from numpy.linalg import LinAlgError

from Configs import Configs
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.MatrixInit import MatrixInit


# https://github.com/wqren/RNN-theano/blob/master/utils/initialization.py

class EsnInit(MatrixInit):
    def __init__(self, scale=0.5, sparsity=0.9, seed=Configs.seed):
        # random generator
        self.__rng = numpy.random.RandomState(seed)
        self.__scale = scale
        self.__sparsity = sparsity

    def init_matrix(self, size, dtype):
        n = size[0]
        m = size[1]

        assert n == m
        trials = 0
        success = False
        while not success:
            values = self.__rng.uniform(low=-1, high=1, size=(n * m,))
            positions = self.__rng.permutation(n * m)
            limit = int(n * n * self.__sparsity)
            if n < 30:
                limit = n * n - n
            values[positions[:limit]] = 0.
            values = values.reshape((n, n))
            try:
                rho = numpy.max(numpy.abs(numpy.linalg.eigvals(values)))
                values = values * self.__scale / rho
                success = True
            except LinAlgError as e:
                print('ESN weights generation, trail', e)
                trials += 1
                if trials > 20:
                    raise ValueError('Could not generate ESN weights')

        return values.astype(dtype=dtype)

    @property
    def infos(self):
        return InfoGroup('ESN init strategy', InfoList(PrintableInfoElement('scale', ':2.2f', self.__scale),
                                                       PrintableInfoElement('sparsity', ':2.2f', self.__sparsity)))

if __name__ == '__main__':
    n = 100
    size = (n, n)
    print('Generating a matrix with size: {} with the ESN initialization ')
    strategy = EsnInit(scale=1, sparsity=0.8)
    w = strategy.init_matrix(size, dtype='float32')
    rho = numpy.max(numpy.abs(numpy.linalg.eigvals(w)))
    print('rho:', rho)
    print('Done...')
    print(w)
