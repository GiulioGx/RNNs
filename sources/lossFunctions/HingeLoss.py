from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class HingeLoss(LossFunction):

    def value(self, y, t):
        s = 1 - t[-1, :, :] * y[-1, :, :]
        return TT.switch(s > 0, s, 0).sum(axis=0).mean()

    def infos(self):
        return SimpleDescription("hinge_loss")
