from Configs import Configs
from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT

class NullLoss(LossFunction):

    def value(self, y, t):
        return ((t[-1:, :, :] - y[-1:, :, :]) ** 2).sum(axis=0).mean()*0 # TODO fixme disconnected

    @property
    def infos(self):
        return SimpleDescription('null_loss')
