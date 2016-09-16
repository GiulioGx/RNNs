import numpy

from Configs import Configs
from combiningRule.LinearCombination import LinearCombination
from infos.InfoGroup import InfoGroup
import theano.tensor as TT
import theano as T
from theano.tensor.shared_randomstreams import RandomStreams

__author__ = 'giulio'


class UniformRandomCombination(LinearCombination):
    def get_linear_coefficients(self, H):
        n = H.shape[0]
        u = self.__srng.uniform(low=0, high=1, size=(n, 1))
        return u

    def __init__(self, normalize_components: bool = True, seed=Configs.seed):
        super().__init__(normalize_components=normalize_components)
        self.__srng = RandomStreams(seed=seed)

    @property
    def infos(self):
        return InfoGroup('random_uniform_combination', super(UniformRandomCombination, self).infos)


if __name__ == '__main__':
    n = TT.scalar('n', dtype='int32')
    combination = UniformRandomCombination(seed=14, normalize_components=False)

    betas = combination.get_linear_coefficients(TT.zeros((n, 1)))
    f = T.function([n], [betas])

    betas_numpy = f(200)
    print('betas', betas_numpy)
    print('sum: ', numpy.sum(betas_numpy))
