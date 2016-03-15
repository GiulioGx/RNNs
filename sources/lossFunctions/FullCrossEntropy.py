from infos.InfoElement import SimpleDescription, PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class FullCrossEntropy(LossFunction):
    def __init__(self, single_probability_ouput: bool = False):
        super().__init__()
        self.__single_probability_output = single_probability_ouput

    def value(self, y, t, mask):

        if self.__single_probability_output:
            c = -(t * TT.log(y + 1e-10) + (1 - t) * TT.log((1 - y) + 1e-10))
        else:
            c = -(t * TT.log(y))
        n_selected_temporal_losses = TT.switch(mask.sum(axis=1) > 0, 1, 0).sum().sum()
        s = (c * mask).sum().sum().sum() / n_selected_temporal_losses
        return s
        # n_selected_temporal_losses = TT.switch(mask.sum(axis=1) > 0, 1, 0).sum(axis=1)
        # s = (c * mask).sum(axis=2).sum(axis=1) / TT.switch(n_selected_temporal_losses>0, n_selected_temporal_losses, 1)
        # return s.sum()
        # s = (c * mask).sum(axis=2).sum(axis=1) / LossFunction.num_examples_insting_temp_loss(mask)
        # return s.sum()

    @property
    def infos(self):
        return InfoGroup('full_cross_entropy_loss', InfoList(PrintableInfoElement('single_prob_output', '',
                                                                                  self.__single_probability_output)))
