from configs import Configs
import theano as T
import theano.tensor as TT
import numpy
import abc

__author__ = 'giulio'


class Penalty:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def penalty_term(self, deriv__a, W_rec):
        """return a theano expression for a penalty term and it's gradient"""
        return


class NullPenalty(Penalty):

    def __init__(self):
        super().__init__()

    def penalty_term(self, deriv__a, W_rec):
        print('null penalty called')

        penalty = TT.alloc(numpy.array(0., dtype=Configs.floatType))
        penalty_grad = TT.zeros_like(W_rec, dtype=Configs.floatType)

        return penalty, penalty_grad


class MeanPenalty(Penalty):

    def __init__(self):
        super().__init__()

    def penalty_term(self, deriv__a, W_rec):
        # deriv__a is a matrix n_steps * n_hidden * n_examples

        mean_deriv_a = deriv__a.mean(axis=2)

        n_steps = mean_deriv_a.shape[0]
        n_hidden = W_rec.shape[0]

        penalty, penalty_grad = penalty_step(mean_deriv_a,
                                             TT.alloc(numpy.array(0., dtype=Configs.floatType)),
                                             TT.eye(n_hidden, dtype=Configs.floatType), W_rec)

        return penalty / n_steps, penalty_grad / n_steps


class FullPenalty(Penalty):
    # very slow
    def __init__(self):
        super().__init__()

    def penalty_term(self, deriv_a, W_rec):
        # deriv__a is a matrix n_steps * n_hidden * n_examples

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

        return penalties[-1] / (n * n_steps), gradients[-1] / (n * n_steps)


# util functions

def penalty_step(deriv_a, penalty_acc, penalty_grad_acc, W_rec):
    # a is a matrix n_steps * n_hidden

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

    A = values[-1]
    penalty = penalty_acc + (1 / (A ** 2).sum())  # 1/frobenius norm
    penalty_grad = penalty_grad_acc + TT.grad(penalty, [W_rec], consider_constant=[deriv_a])[0]

    return penalty, penalty_grad
