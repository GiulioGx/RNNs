from infos.InfoElement import SimpleDescription
from lossFunctions.LossFunction import LossFunction


class SquaredError(LossFunction):

    def value(self, y, t):
        return ((t[-1:, :, :] - y[-1:, :, :]) ** 2).sum(axis=0).mean()

    def infos(self):
        return SimpleDescription('squared_error')
