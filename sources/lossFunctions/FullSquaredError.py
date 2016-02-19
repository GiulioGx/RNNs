from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction


class FullSquaredError(LossFunction):
    def value(self, y, t):
        return (((t - y) * self.mask) ** 2).sum(axis=0).sum(axis=0).mean(axis=0)

    @property
    def infos(self):
        return SimpleDescription('full squared_error_loss')
