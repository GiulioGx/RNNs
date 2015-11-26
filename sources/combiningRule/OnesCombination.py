from Configs import Configs
from combiningRule.LinearCombination import LinearCombination
import theano.tensor as TT
import theano as T
from infos.InfoElement import SimpleDescription
from infos.InfoGroup import InfoGroup
from theanoUtils import norm, is_inf_or_nan, normalize

__author__ = 'giulio'


class OnesCombination(LinearCombination):

    def __init__(self, normalize_components: bool = True):
        super().__init__(normalize_components=normalize_components)

    def get_linear_coefficients(self, vector_list, n):
        return TT.ones((n, 1), dtype=Configs.floatType)

    @property
    def infos(self):
        return InfoGroup('ones_combination', super(OnesCombination, self).infos)
