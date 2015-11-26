from Configs import Configs
from combiningRule.LinearCombination import LinearCombination
from infos.InfoElement import SimpleDescription
from infos.InfoGroup import InfoGroup
from theanoUtils import norm, is_inf_or_nan, normalize
import theano.tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams

__author__ = 'giulio'


class SimplexCombination(LinearCombination):
    def get_linear_coefficients(self, vector_list, n):
        u = self.__srng.uniform(low=0, high=1, size=(n, 1))
        # x = TT.exp(1.-u)
        # r = x/x.sum()
        r = u / u.sum()  # XXX simplex
        return r

    def __init__(self, normalize_components: bool = True, seed=Configs.seed):
        super().__init__(normalize_components=normalize_components)
        self.__srng = RandomStreams(seed=seed)

    @property
    def infos(self):
        return InfoGroup('simplex_combination', super(SimplexCombination, self).infos)
