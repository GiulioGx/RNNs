from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class FullCrossEntropy(LossFunction):

    def __init__(self, single_probability_ouput:bool = False):
        self.__single_probability_output = single_probability_ouput

    def value(self, y, t, mask):
        if self.__single_probability_output:
            return (-(t * TT.log(y) + (1 - t) * TT.log(1 - y))*mask).sum(axis=1).mean(axis=0).mean(axis=0)
        else:
            return -((t * TT.log(y))*mask).sum(axis=0).sum(axis=0).mean(axis=0)

    @property
    def infos(self):
        return SimpleDescription('full_cross_entropy_loss')
