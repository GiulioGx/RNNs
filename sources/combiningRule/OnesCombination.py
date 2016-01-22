from Configs import Configs
from combiningRule.LinearCombination import LinearCombination
import theano.tensor as TT
from infos.InfoGroup import InfoGroup

__author__ = 'giulio'


class OnesCombination(LinearCombination):
    def __init__(self, normalize_components: bool = True):
        super().__init__(normalize_components=normalize_components)

    def get_linear_coefficients(self, H):
        n = H.shape[0]
        return TT.ones((n, 1), dtype=Configs.floatType)

    @property
    def infos(self):
        return InfoGroup('ones_combination', super(OnesCombination, self).infos)
