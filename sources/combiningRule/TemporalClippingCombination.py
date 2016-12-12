from theano.tensor.shared_randomstreams import RandomStreams

from Configs import Configs
from combiningRule import LinearCombination
from infos.InfoGroup import InfoGroup
import theano.tensor as TT

from theanoUtils import tensor_median


class TemporalClippingCombination(LinearCombination):
    def __init__(self, thr: float, clip_style: str, seed=Configs.seed):
        super().__init__(normalize_components=False)
        self.__clip_thr = thr
        self.__srng = RandomStreams(seed=seed)

    @property
    def infos(self):
        return InfoGroup('temporal_clipping_combination', super(TemporalClippingCombination, self).infos)

    def get_linear_coefficients(self, H):
        n = H.shape[0]

        temporal_norms = H.norm(2, axis=1).reshape((H.shape[0], 1))
        median = temporal_norms[TT.cast(n/4, dtype='int32')]

        coefficients = TT.switch(temporal_norms < median, 1., median / temporal_norms)

        return coefficients
