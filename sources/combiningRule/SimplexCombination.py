from Configs import Configs
from combiningRule.CombiningRule import CombiningRule
from theanoUtils import norm, is_not_real
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams


__author__ = 'giulio'


class SimplexCombination(CombiningRule):

    def get_linear_coefficients(self, vector_list, n):
        b = self.__srng.exponential(scale=1, size=(1, n))
        return b/b.sum()

    def __init__(self, seed=Configs.seed):
        self.__srng = RandomStreams(seed=seed)

    def normalize_step(self, grads_combinantion, norms):
        return grads_combinantion/norm(grads_combinantion)

