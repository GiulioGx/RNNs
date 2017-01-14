from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.MatrixInit import MatrixInit
import numpy as np
from initialization.OrtoghonalInit import OrtoghonalInit

__author__ = 'giulio'


class EigenInit(MatrixInit):
    """Initialize a matrix with eigenvalues with given mean and std_dev"""

    def __init__(self, seed: int, mean: float = 1., std_dev: float = 0.01):
        # random generator
        self.__rnd = np.random.RandomState(seed)
        self.__mean = mean
        self.__std_dev = std_dev
        self.__seed = seed

    def init_matrix(self, size, dtype):
        q = OrtoghonalInit(seed=self.__seed).init_matrix(size=size, dtype=dtype)
        diagonal = self.__rnd.normal(loc=self.__mean, scale=self.__std_dev, size=(size[0]))
        d = np.diag(diagonal)
        result = np.dot(np.dot(q, d), np.transpose(q))

        # experimental
        u, _, v = np.linalg.svd(self.__rnd.normal(loc=0, scale=self.__std_dev, size=size))
        # result = np.dot(np.dot(u, d), v)

        # experimental 2
        # result = d


        # stats
        eig_values = np.abs(np.linalg.eigvals(result))
        cond = max(eig_values) / min(eig_values)

        print("std: {:.2f}, mean: {:.2f}, cond:{:.2f}, std_eig:  {:.2f}".format(np.std(result), np.mean(result), cond,
                                                                                np.std(np.array(eig_values))))
        return result.astype(dtype)

    @property
    def infos(self):
        return InfoGroup('eigen init strategy', InfoList(PrintableInfoElement('mean', ':2.2f', self.__mean),
                                                         PrintableInfoElement('std_dev', ':.2f', self.__std_dev)))


if __name__ == '__main__':
    n = 5
    size = (n, n)
    print('Generating a matrix with size: {} with the eigen initialization strategy')
    strategy = EigenInit(mean=1, std_dev=0.1, seed=14)
    w = strategy.init_matrix(size, dtype='float32')
    print(w)
    print('generated matrix is\n', w)
    print('spectral radius of the generated matrix is {:2.2f}'.format(MatrixInit.spectral_radius(w)))
    print('Done...')
