import numpy

from Configs import Configs
from combiningRule.LinearCombination import LinearCombination
from infos.InfoGroup import InfoGroup
import theano.tensor as TT
import theano as T
from theano.tensor.shared_randomstreams import RandomStreams

__author__ = 'giulio'


class SimplexCombination(LinearCombination):
    def get_linear_coefficients(self, H):
        n = H.shape[0]
        u = self.__srng.uniform(low=0, high=1, size=(n, 1))
        x = TT.exp(1. - u)
        r = x / (x.sum() + 1e-10)
        # r = u / u.sum()  # XXX simplex
        return r

    def __init__(self, normalize_components: bool = True, seed=Configs.seed):
        super().__init__(normalize_components=normalize_components)
        self.__srng = RandomStreams(seed=seed)

    @property
    def infos(self):
        return InfoGroup('simplex_combination', super(SimplexCombination, self).infos)


if __name__ == '__main__':
    n = TT.scalar('n', dtype='int32')
    combination = SimplexCombination(seed=14, normalize_components=False)  # XXX probably broken

    betas = combination.get_linear_coefficients(TT.zeros((n, 1)))
    f = T.function([n], [betas])

    betas_numpy = f(200)
    print('betas', betas_numpy)
    print('sum: ', numpy.sum(betas_numpy))
