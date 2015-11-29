from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class CrossEntropy(LossFunction):
    def value(self, y, t):
        # return TT.log(y[-1, 0, :])
        return -(t[-1, :, :] * TT.log(y[-1, :, :])).mean(axis=1).sum()
        # return (-t[-1, 0, :] * TT.log(y[-1, 0, :]) - (1. - t[-1, 0, :]) * TT.log(1. - y[-1, 0, :])).mean()  # FIXME

    def infos(self):
        return SimpleDescription('cross_entropy')
