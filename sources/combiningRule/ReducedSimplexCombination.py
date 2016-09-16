import numpy
import theano as T
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams

from Configs import Configs
from combiningRule.LinearCombination import LinearCombination
from infos.InfoGroup import InfoGroup

__author__ = 'giulio'


class ReducedSimplexCombination(LinearCombination):
    def get_linear_coefficients(self, H):
        n = H.shape[0]

        if self.__keep_ratio < 1:

            p = TT.cast(n * self.__keep_ratio, dtype='int32')
            indexes = self.__srng.choice(size=(p, 1), a=TT.arange(0, n, 1), replace=False, p=None, dtype='int32')
            c = TT.zeros(shape=(n, 1), dtype=Configs.floatType)
        else:
            p = n

        u = self.__srng.uniform(low=0, high=1, size=(p, 1))
        x = TT.exp(1. - u)
        r = x / (x.sum() + 1e-10)

        if self.__keep_ratio < 1:

            c = TT.set_subtensor(c[indexes,0], r)
        else:
            c = r

        # r = u / u.sum()  # XXX simplex
        return c

    def __init__(self, keep_ratio: float, normalize_components: bool = True, seed=Configs.seed):
        super().__init__(normalize_components=normalize_components)
        self.__srng = RandomStreams(seed=seed)
        assert (0 < keep_ratio <= 1)
        self.__keep_ratio = keep_ratio

    @property
    def infos(self):
        return InfoGroup('reduced_simplex_combination', super(ReducedSimplexCombination, self).infos)


if __name__ == '__main__':
    n = TT.scalar('n', dtype='int32')
    combination = ReducedSimplexCombination(keep_ratio=0.5, seed=13, normalize_components=False)  # XXX probably broken

    betas = combination.get_linear_coefficients(TT.zeros((n, 1)))
    f = T.function([n], [betas])

    betas_numpy = f(10)
    print('betas', betas_numpy)
    print('sum: ', numpy.sum(betas_numpy))
