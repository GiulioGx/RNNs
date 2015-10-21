from Configs import Configs
import theano as T
import theano.tensor as TT
import numpy
import abc
from InfoProducer import InfoProducer
from theanoUtils import norm

__author__ = 'giulio'


class PenaltySymbols(InfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def penalty_value(self):
        """returns a theano expression for penalty_value """

    @abc.abstractmethod
    def penalty_grad(self):
        """returns a theano expression for penalty_grad """


class Penalty(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, W_rec, W_in, W_out, b_rec, b_out, net_symbols):
        """returns the compiled version"""


class NullPenalty(Penalty):
    class Symbols(PenaltySymbols):
        def __init__(self, W_rec):
            self.__penalty_value = TT.alloc(numpy.array(0., dtype=Configs.floatType))
            self.__penalty_grad = TT.zeros_like(W_rec, dtype=Configs.floatType)

        @property
        def penalty_grad(self):
            return self.__penalty_grad

        @property
        def penalty_value(self):
            return self.__penalty_value

        def format_infos(self, infos):
            return '', []

        @property
        def infos(self):
            return []

    def __init__(self):
        super().__init__()

    def compile(self, W_rec, W_in, W_out, b_rec, b_out, net_symbols):
        return NullPenalty.Symbols(W_rec)


class FullPenalty(Penalty):
    class Symbols(PenaltySymbols):
        def __init__(self, W_rec, W_in, W_out, b_rec, b_out, net_symbols):
            # deriv__a is a matrix n_steps * n_hidden * n_examples

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

        def format_infos(self, infos):
            return 'penalty=[value: {:07.3f}, grad: {:07.3f}]'.format(infos[0].item(), infos[1].item()), infos[
                                                                                                         2:len(
                                                                                                             infos)]

        @property
        def infos(self):
            return self.__infos

    # very slow
    def __init__(self):
        super().__init__()

    def compile(self, W_rec, W_in, W_out, b_rec, b_out, net_symbols):
        return FullPenalty.Symbols(W_rec, W_in, W_out, b_rec, b_out, net_symbols)


class MeanPenalty(Penalty):
    class Symbols(PenaltySymbols):
        def __init__(self, W_rec, W_in, W_out, b_rec, b_out, net_symbols):
            # deriv__a is a matrix n_steps * n_hidden * n_examples

            deriv_a = net_symbols.get_deriv_a(W_rec, W_in, W_out, b_rec, b_out)

            mean_deriv_a = deriv_a.mean(axis=2)

            n_steps = mean_deriv_a.shape[0]
            n_hidden = W_rec.shape[0]

            penalty_value, penalty_grad = penalty_step(mean_deriv_a,
                                                       TT.alloc(numpy.array(0., dtype=Configs.floatType)),
                                                       TT.eye(n_hidden, dtype=Configs.floatType), W_rec)

            self.__penalty_value = TT.cast(penalty_value / n_steps, dtype=Configs.floatType)
            self.__penalty_grad = TT.cast(penalty_grad / n_steps, dtype=Configs.floatType)

            self.__infos = [self.__penalty_value, norm(self.__penalty_grad)]

        @property
        def penalty_grad(self):
            return self.__penalty_grad

        @property
        def penalty_value(self):
            return self.__penalty_value

        def format_infos(self, infos):
            return 'penalty=[value: {:07.3f}, grad: {:07.3f}]'.format(infos[0].item(), infos[1].item()), infos[
                                                                                                         2:len(
                                                                                                             infos)]

        @property
        def infos(self):
            return self.__infos

    def __init__(self):
        super().__init__()

    def compile(self, W_rec, W_in, W_out, b_rec, b_out, net_symbols):
        return MeanPenalty.Symbols(W_rec, W_in, W_out, b_rec, b_out, net_symbols)


class ConstantPenalty(Penalty):
    class Symbols(PenaltySymbols):
        def __init__(self, W_rec, W_in, W_out, b_rec, b_out, net_symbols):
            # deriv__a is a matrix n_steps * n_hidden * n_examples

            deriv_a = net_symbols.get_deriv_a(W_rec, W_in, W_out, b_rec, b_out)

            mean_deriv_a = deriv_a.mean(axis=2)
            #n_steps = TT.cast(mean_deriv_a.shape[0], dtype=Configs.floatType)

            A = deriv_a_T_wrt_a1(W_rec, mean_deriv_a)

            self.__penalty_value = ((A ** 2).sum() - 1) ** 2
            self.__penalty_grad = TT.grad(self.__penalty_value, [W_rec], consider_constant=[deriv_a])[0]

            self.__infos = [self.__penalty_value, norm(self.__penalty_grad)]

        @property
        def penalty_grad(self):
            return self.__penalty_grad

        @property
        def penalty_value(self):
            return self.__penalty_value

        def format_infos(self, infos):
            return 'penalty=[value: {:07.3f}, grad: {:07.3f}]'.format(infos[0].item(), infos[1].item()), infos[
                                                                                                         2:len(
                                                                                                             infos)]

        @property
        def infos(self):
            return self.__infos

    def compile(self, W_rec, W_in, W_out, b_rec, b_out, net_symbols):
        return ConstantPenalty.Symbols(W_rec, W_in, W_out, b_rec, b_out, net_symbols)


# util functions

def penalty_step(deriv_a, penalty_acc, penalty_grad_acc, W_rec):
    # a is a matrix n_steps * n_hidden

    A = deriv_a_T_wrt_a1(W_rec, deriv_a)
    penalty = penalty_acc + (1 / (A ** 2).sum())  # 1/frobenius norm
    penalty_grad = penalty_grad_acc + TT.grad(penalty, [W_rec], consider_constant=[deriv_a])[0]

    return penalty, penalty_grad


def deriv_a_T_wrt_a1(W_rec, deriv_a):
    def prod_step(deriv_a_t, A_prev, W_rec):
        A = TT.dot(A_prev, (W_rec * deriv_a_t))
        return A

    n_hidden = W_rec.shape[0]
    values, _ = T.scan(prod_step, sequences=deriv_a,
                       outputs_info=[TT.eye(n_hidden, dtype=Configs.floatType)],
                       go_backwards=True,
                       non_sequences=[W_rec],
                       name='prod_step',
                       mode=T.Mode(linker='cvm'))

    return values[-1]
