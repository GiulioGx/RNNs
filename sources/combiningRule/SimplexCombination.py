from Configs import Configs
from combiningRule.LinearCombination import LinearCombinationRule
from theanoUtils import norm, is_not_real
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams


__author__ = 'giulio'


class SimplexCombination(LinearCombinationRule):

    def get_linear_coefficients(self, vector_list, n):

        u = self.__srng.uniform(low=0, high=1, size=(n, 1))
        x = TT.exp(1.-u)
        r = x/x.sum()
        return r

    def __init__(self, seed=Configs.seed):
        self.__srng = RandomStreams(seed=seed)

    def normalize_step(self, grads_combinantion, norms):
        norm_comb = grads_combinantion.norm(2)
        return TT.switch(norm_comb <= 0, grads_combinantion, grads_combinantion/norm_comb)

