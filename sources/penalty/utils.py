from theano import tensor as TT
import theano as T
from Configs import Configs

__author__ = 'giulio'


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