from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction
import theano.tensor as TT


class CrossEntropy(LossFunction):  # XXX deprecated
    def value(self, y, t, mask):
        return -(t * TT.log(y)*mask)[-1].mean(axis=1).sum()

    @property
    def infos(self):
        return SimpleDescription('cross_entropy_loss')
