from Configs import Configs
import theano as T
import theano.tensor as TT
import numpy
import abc
from theanoUtils import norm

__author__ = 'giulio'


class Penalty(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def penalty_term(self, W_rec, W_in, W_out, b_rec, b_out, symbolic_closet):
        """return a theano expression for a penalty term and it's gradient"""
        return


class NullPenalty(Penalty):
    def __init__(self):
        super().__init__()

    def penalty_term(self, W_rec, W_in, W_out, b_rec, b_out, symb_closet):
        penalty = TT.alloc(numpy.array(0., dtype=Configs.floatType))
        penalty_grad = TT.zeros_like(W_rec, dtype=Configs.floatType)

        return penalty, penalty_grad


class FullPenalty(Penalty):
    # very slow
    def __init__(self):
        super().__init__()

    def penalty_term(self, W_rec, W_in, W_out, b_rec, b_out, symbolic_closet):
        # deriv__a is a matrix n_steps * n_hidden * n_examples

        deriv_a = symbolic_closet.get_deriv_a(W_rec, W_in, W_out, b_rec, b_out)
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

        penalty_grad = TT.cast(gradients[-1] / (n * n_steps), dtype=Configs.floatType)
        penalty_value = TT.cast(penalties[-1] / (n * n_steps), dtype=Configs.floatType)

        return penalty_value, penalty_grad


class MeanPenalty(Penalty):
    def __init__(self):
        super().__init__()

    def penalty_term(self, W_rec, W_in, W_out, b_rec, b_out, symbolic_closet):
        # deriv__a is a matrix n_steps * n_hidden * n_examples

        deriv_a = symbolic_closet.get_deriv_a(W_rec, W_in, W_out, b_rec, b_out)

        mean_deriv_a = deriv_a.mean(axis=2)

        n_steps = mean_deriv_a.shape[0]
        n_hidden = W_rec.shape[0]

        penalty_value, penalty_grad = penalty_step(mean_deriv_a,
                                                   TT.alloc(numpy.array(0., dtype=Configs.floatType)),
                                                   TT.eye(n_hidden, dtype=Configs.floatType), W_rec)

        penalty_value = TT.cast(penalty_value / n_steps, dtype=Configs.floatType)
        penalty_grad = TT.cast(penalty_grad / n_steps, dtype=Configs.floatType)

        return penalty_value, penalty_grad


class ConstantPenalty(Penalty):

    def penalty_term(self, W_rec, W_in, W_out, b_rec, b_out, symbolic_closet):
        # deriv__a is a matrix n_steps * n_hidden * n_examples

        deriv_a = symbolic_closet.get_deriv_a(W_rec, W_in, W_out, b_rec, b_out)

        mean_deriv_a = deriv_a.mean(axis=2)
        n_steps = TT.cast(mean_deriv_a.shape[0], dtype=Configs.floatType)

        A = deriv_a_T_wrt_a1(W_rec, mean_deriv_a)
        penalty_value = ((A ** 2).sum() - 1) ** 2   # 1/frobenius norm
        penalty_grad = TT.grad(penalty_value, [W_rec], consider_constant=[deriv_a])[0]

        penalty_value = norm(A)

        return penalty_value,  penalty_grad


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
