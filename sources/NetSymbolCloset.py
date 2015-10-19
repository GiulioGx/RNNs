import theano as T
import theano.tensor as TT
from Configs import Configs

__author__ = 'giulio'


class NetSymbolCloset:
    def __init__(self, net, W_rec, W_in, W_out, b_rec, b_out):
        self.__net = net

        # define shared variables
        self.__W_in = T.shared(W_in, 'W_in')
        self.__W_rec = T.shared(W_rec, 'W_rec')
        self.__W_out = T.shared(W_out, 'W_out')
        self.__b_rec = T.shared(b_rec, 'b_rec', broadcastable=(False, True))
        self.__b_out = T.shared(b_out, 'b_out', broadcastable=(False, True))

        # define symbols
        W_in = TT.matrix()
        W_rec = TT.matrix()
        W_out = TT.matrix()
        b_rec = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True))
        b_out = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True))

        self.u = TT.tensor3()  # input tensor
        self.t = TT.tensor3()  # label tensor

        self.y, self.deriv_a = net.net_output(W_rec, W_in, W_out, b_rec, b_out, self.u)
        self.y_shared, self.deriv_a_shared = T.clone([self.y, self.deriv_a],
                                                     replace={W_rec: self.__W_rec, W_in: self.__W_in,
                                                              W_out: self.__W_out, b_rec: self.__b_rec,
                                                              b_out: self.__b_out})

    def get_deriv_a(self, W_rec, W_in, W_out, b_rec, b_out):
        _, deriv_a = self.__net.net_output(W_rec, W_in, W_out, b_rec, b_out, self.u)
        return deriv_a


    #     # define separate gradients
    #     self.__W_fix = self.__W_rec.clone()
    #
    # def net_output(self, W_rec, W_in, W_out, b_rec, b_out, u, W_fix, target_index):
    #     n_sequences = u.shape[2]
    #     h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.__net.n_hidden, n_sequences)
    #
    #     values, _ = T.scan(self.net_output_t, sequences=u,
    #                        outputs_info=[h_m1, 0, None, None],
    #                        non_sequences=[W_rec, W_in, W_out, b_rec, b_out, W_fix, target_index],
    #                        name='net_output',
    #                        mode=T.Mode(linker='cvm'))
    #     y = values[2]
    #     deriv_a = values[3]
    #     return y, deriv_a
    #
    # def net_output_t(self, u_t, h_tm1, i, W_rec, W_in, W_out, b_rec, b_out, W_fix, target_index):
    #     W = TT.switch(TT.eq(i, target_index), W_fix, W_rec)
    #     h_t, deriv_a_t = self.__net.h_t(u_t, h_tm1, W, W_in, b_rec)
    #     y_t = self.__net.y_t(h_t, W_out, b_out)
    #     return h_t, i + 1, y_t, deriv_a_t
    #
    # def get_separate(self, target_index):
    #     y_f, _ = self.net_output(self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out, self.u,
    #                                    self.__W_fix, target_index)
    #     loss_f = self.__loss_fnc(y_f, self.t)
    #
    #     gW_rec, gW_in, gW_out, \
    #     gb_rec, gb_out = TT.grad(loss_f,
    #                                        [self.__W_fix, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
    #     grad_norm = TT.sqrt((gW_rec ** 2).sum() +
    #                              (gW_in ** 2).sum() +
    #                              (gW_out ** 2).sum() +
    #                              (gb_rec ** 2).sum() +
    #                              (gb_out ** 2).sum())
    #
    #     return gW_rec, gW_in, gW_out, gb_rec, gb_out

    @property
    def W_rec(self):
        return self.__W_rec

    @property
    def W_in(self):
        return self.__W_in

    @property
    def W_out(self):
        return self.__W_out

    @property
    def b_rec(self):
        return self.__b_rec

    @property
    def b_out(self):
        return self.__b_out
