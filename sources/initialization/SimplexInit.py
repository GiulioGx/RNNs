from Configs import Configs
from infos.InfoElement import SimpleDescription
from initialization.MatrixInit import MatrixInit
import numpy

__author__ = 'giulio'


class SimplexInit(MatrixInit):
    def __init__(self, columnwise: bool = True, seed=Configs.seed):
        # random generator
        self.__rng = numpy.random.RandomState(seed)
        self.__columnwise = columnwise

    def init_matrix(self, size, dtype):
        # TODO assert inputs

        if not self.__columnwise:
            size = (size[1], size[0])

        u = self.__rng.uniform(low=0, high=1, size=size).astype(Configs.floatType)
        # x = TT.exp(1.-u)
        # r = x/x.sum()
        w = u / u.sum(axis=0)  # XXX simplex

        if self.__columnwise:
            return w
        else:
            return numpy.transpose(w)

    @property
    def infos(self):
        return SimpleDescription('simplex init strategy')


if __name__ == '__main__':
    size = (5, 7)
    print('Generating a matrix with size: {} with the simplex initialization '
          'strategy columnwise (columns add up to one)...'.format(size))
    strategy = SimplexInit(columnwise=True)
    w = strategy.init_matrix(size, dtype='float32')
    print('Done...')
    print(w)
    print('##########')
    print('Generating a matrix with size: {} with the simplex initialization'
          ' strategy rowwise (rows add up to one)...'.format(size))
    strategy = SimplexInit(columnwise=False)
    w = strategy.init_matrix(size, dtype='float32')
    print('Done...')
    print(w)
