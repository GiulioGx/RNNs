from Configs import Configs
from combiningRule.LinearCombination import LinearCombinationRule
from infos.InfoElement import SimpleDescription
from theanoUtils import norm, is_not_real, normalize
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams


__author__ = 'giulio'


class SimplexCombination(LinearCombinationRule):

    def get_linear_coefficients(self, vector_list, n):

        u = self.__srng.uniform(low=0, high=1, size=(n, 1))
        # x = TT.exp(1.-u)
        # r = x/x.sum()
        r = u/u.sum()  # XXX simplex
        return r

    def __init__(self, seed=Configs.seed):
        self.__srng = RandomStreams(seed=seed)

    def normalize_step(self, grads_combinantion, norms):
        return normalize(grads_combinantion)
        #return grads_combinantion

    @property
    def infos(self):
        return SimpleDescription('simplex_combination')

