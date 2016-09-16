import numpy

from Configs import Configs
from combiningRule.LinearCombination import LinearCombination
from infos.InfoGroup import InfoGroup
import theano.tensor as TT
import theano as T
from theano.tensor.shared_randomstreams import RandomStreams

__author__ = 'giulio'


class TimeSmoothingCombination(LinearCombination):
    def get_linear_coefficients(self, H):
        n = H.shape[0]

        x = TT.arange(1, n+1, 1)

        return 1. / TT.exp(0.005*x)
        #return n - x

    def __init__(self, normalize_components: bool = True, seed=Configs.seed):
        super().__init__(normalize_components=normalize_components)
        self.__srng = RandomStreams(seed=seed)

    @property
    def infos(self):
        return InfoGroup('time_smoothing_combination', super(TimeSmoothingCombination, self).infos)
