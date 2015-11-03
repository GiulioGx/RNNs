from Configs import Configs
from combiningRule.CombiningRule import CombiningRule
import theano.tensor as TT
import theano as T
from theanoUtils import norm, is_not_real

__author__ = 'giulio'


class NormalizedSum(CombiningRule):

    def normalize_step(self, grads_combination, norms):
        return grads_combination/norm(grads_combination)

    def get_linear_coefficients(self, vector_list, n):
        return TT.ones((n, 1), dtype=Configs.floatType)




