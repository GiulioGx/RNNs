from theano.tensor.shared_randomstreams import RandomStreams

from Configs import Configs
from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class SimplexCrossEntropy(LossFunction):
    def __init__(self, single_probability_ouput: bool = False):
        super().__init__()
        self.__single_probability_output = single_probability_ouput
        self.__srng = RandomStreams(seed=Configs.seed)
        self.__drop_rate = 0.7

    def value(self, y, t, mask):

        if self.__single_probability_output:
            c = -(t * TT.log(y + 1e-10) + (1 - t) * TT.log((1 - y) + 1e-10))
        else:
            c = -(t * TT.log(y))

        d = c * mask
        d = d.mean(axis=2).sum(axis=1)
        d = d * self.__srng.choice(size=d.shape, a=[0, 1], replace=True, p=[self.__drop_rate, 1 - self.__drop_rate],
                                     dtype=Configs.floatType)

        return d.mean()

    @property
    def infos(self):
        return InfoGroup('simplex_cross_entropy_loss', InfoList(PrintableInfoElement('single_prob_output', '',
                                                                                     self.__single_probability_output)))
