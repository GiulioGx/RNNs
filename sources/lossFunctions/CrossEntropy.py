from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class CrossEntropy(LossFunction):
    def value(self, y, t):
        return -(t[-1, :, :] * TT.log(y[-1, :, :])).mean(axis=1).sum()

    @property
    def infos(self):
        return SimpleDescription('cross_entropy_loss')
