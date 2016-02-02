from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class FullCrossEntropy(LossFunction):
    def value(self, y, t):
        return -(t[:, :, :] * TT.log(y[:, :, :]) + (1 - t[:, :, :]) * TT.log(1 - y[:, :, :])).sum(axis=1).mean().mean()

    @property
    def infos(self):
        return SimpleDescription('full_cross_entropy_loss')
