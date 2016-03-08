from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction


class SquaredError(LossFunction): # XXX deprecated use FullSquared with mask instead

    def value(self, y, t, mask):
        return (((t-y) * mask)[-1] ** 2).sum().sum().mean()

    @property
    def infos(self):
        return SimpleDescription('squared_error_loss')
