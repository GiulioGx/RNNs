from Configs import Configs
from combiningRule.LinearCombination import LinearCombinationRule
import theano.tensor as TT
import theano as T

from infos.InfoElement import SimpleDescription
from theanoUtils import norm, is_not_real, normalize

__author__ = 'giulio'


class NormalizedSum(LinearCombinationRule):

    def normalize_step(self, grads_combination, norms):
        return normalize(grads_combination)

    def get_linear_coefficients(self, vector_list, n):
        return TT.ones((n, 1), dtype=Configs.floatType)

    @property
    def infos(self):
        return SimpleDescription('normalized_sum_combination')




