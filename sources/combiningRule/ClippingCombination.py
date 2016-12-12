from theano.tensor.shared_randomstreams import RandomStreams

from Configs import Configs
from combiningRule import LinearCombination
from infos.InfoGroup import InfoGroup
import theano.tensor as TT


class ClippingCombination(LinearCombination):
    def __init__(self, thr: float, clip_style: str, seed=Configs.seed):
        super().__init__(normalize_components=False)
        self.__clip_thr = thr
        self.__srng = RandomStreams(seed=seed)


    @property
    def infos(self):
        return InfoGroup('clipping_combination', super(ClippingCombination, self).infos)

    def get_linear_coefficients(self, H):
        # n = H.shape[0]

        temporal_norms = H.norm(2, axis=1).reshape((H.shape[0], 1))
        random_thr = self.__srng.uniform(low=TT.min(temporal_norms), high=TT.max(temporal_norms), size=(1,))

        coefficients = TT.switch(temporal_norms < random_thr, 1., random_thr / temporal_norms)


        return coefficients
