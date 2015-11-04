from Configs import Configs
from combiningRule.CombiningRule import CombiningRule
from theanoUtils import norm, is_not_real
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams


__author__ = 'giulio'


class SimplexCombination(CombiningRule):

    def get_linear_coefficients(self, vector_list, n):

        u = self.__srng.uniform(low=0, high=1, size=(n, 1))

        exp_lambda = 1
        x = -TT.log(-u+1)/exp_lambda
        return x/x.sum()

    def __init__(self, seed=Configs.seed):
        self.__srng = RandomStreams(seed=seed)

    def normalize_step(self, grads_combinantion, norms):
        return grads_combinantion/norm(grads_combinantion)

