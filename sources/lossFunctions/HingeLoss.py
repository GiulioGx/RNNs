from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class HingeLoss(LossFunction):
    def value(self, y, t):
        s = 1. - (t * y * self.mask)[:, 0, :]  # XXX defined only for one output unit
        return TT.switch(s > 0, s, 0).mean(axis=1).sum()

    @property
    def infos(self):
        return SimpleDescription("hinge_loss")
