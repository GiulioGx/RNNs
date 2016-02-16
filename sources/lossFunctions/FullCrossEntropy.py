from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class FullCrossEntropy(LossFunction):
    def __init__(self, single_probability_ouput: bool = False):
        self.__single_probability_output = single_probability_ouput

    def value(self, y, t, mask):

        if self.__single_probability_output:
            c = -(t * TT.log(y) + (1 - t) * TT.log(1 - y))
        else:
            c = -(t * TT.log(y))
        n_selected_temporal_losses = TT.switch(mask.norm(2, axis=1) > 0, 1, 0).sum().sum()
        s = (c * mask).sum().sum().sum() / n_selected_temporal_losses
        return s

    @property
    def infos(self):
        return SimpleDescription('full_cross_entropy_loss')  # TODO add single prob option
