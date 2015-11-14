import numpy
import theano as T
from theano import tensor as TT

from Configs import Configs
from infos.InfoElement import PrintableInfoElement, SimpleDescription
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from model.Variables import Variables
from penalty.Penalty import Penalty
from penalty.utils import penalty_step
from theanoUtils import norm

__author__ = 'giulio'


class FullPenalty(Penalty):
    class Symbols(Penalty.Symbols):
        def __init__(self, params, net_symbols):
            # deriv__a is a matrix n_steps * n_hidden * n_examples

            # FIXME
            W_rec = params.W_rec
            W_in = params.W_in
            W_out = params.W_out
            b_rec = params.b_rec
            b_out = params.b_out

            deriv_a = net_symbols.get_deriv_a(W_rec, W_in, W_out, b_rec, b_out)
            permuted_a = deriv_a.dimshuffle(2, 0, 1)
            n_hidden = W_rec.shape[0]

            values, _ = T.scan(penalty_step, sequences=permuted_a,
                               outputs_info=[TT.alloc(numpy.array(0., dtype=Configs.floatType)),
                                             TT.eye(n_hidden, dtype=Configs.floatType)],
                               non_sequences=[W_rec],
                               name='penalty_step',
                               mode=T.Mode(linker='cvm'))

            n = deriv_a.shape[2]
            n_steps = deriv_a.shape[0]
            penalties = values[0]
            gradients = values[1]

            self.__penalty_grad = TT.cast(gradients[-1] / (n * n_steps), dtype=Configs.floatType)
            self.__penalty_value = TT.cast(penalties[-1] / (n * n_steps), dtype=Configs.floatType)

            self.__infos = [self.__penalty_value, norm(self.__penalty_grad)]

        @property
        def penalty_grad(self):
            return self.__penalty_grad

        @property
        def penalty_value(self):
            return self.__penalty_value

        def format_infos(self, info_symbols):
            penalty_value_info = PrintableInfoElement('value', ':07.3f', info_symbols[0].item())
            penalty_grad_info = PrintableInfoElement('value', ':07.3f', info_symbols[1].item())
            info = InfoGroup('penalty', InfoList(penalty_value_info, penalty_grad_info))
            return info, info_symbols[info.length:len(info_symbols)]

        @property
        def infos(self):
            return self.__infos

    # very slow
    def __init__(self):
        super().__init__()

    def compile(self, params: Variables, net_symbols):
        return FullPenalty.Symbols(params, net_symbols)

    @property
    def infos(self):
        return SimpleDescription('full_penalty')
