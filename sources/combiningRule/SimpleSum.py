from Configs import Configs
from combiningRule.CombiningRule import CombiningRule
import theano.tensor as TT

__author__ = 'giulio'


class SimpleSum(CombiningRule):
    def get_linear_coefficients(self, vector_list, n):
        return TT.ones((n, 1), dtype=Configs.floatType)

    def normalize_step(self, grads_combinantion, norms):
        return grads_combinantion
