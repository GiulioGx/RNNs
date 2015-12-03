from infos.InfoElement import SimpleDescription
from output_fncs.OutputFunction import OutputFunction
import theano.tensor as TT


class Softmax(OutputFunction):

    def value(self, x):
        return TT.nnet.softmax(x.T).T

    # alternatively
    #     e_y = TT.exp(y - y.max(axis=0))
    #     return e_y / e_y.sum(axis=0)

    @property
    def infos(self):
        return SimpleDescription('softmax_output_function')
