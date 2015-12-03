from infos.InfoElement import SimpleDescription
from output_fncs.OutputFunction import OutputFunction
import theano.tensor as TT


class Logistic(OutputFunction):

    def value(self, x):
        return 1. / (1. + TT.exp(-x))

    @property
    def infos(self):
        return SimpleDescription('logistic_output_function')
