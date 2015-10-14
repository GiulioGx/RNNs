import theano as T
import theano.tensor as TT
from Configs import Configs

__author__ = 'giulio'


class SymbolsCloset:
    def __init__(self, net, W_rec, W_in, W_out, b_rec, b_out, loss_fnc):
        self.__net = net
        self.__loss_fnc = loss_fnc

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
                                                     replace=[(W_rec, self.__W_rec), (W_in, self.__W_in),
                                                              (W_out, self.__W_out),
                                                              (b_rec, self.__b_rec), (b_out, self.__b_out)])


        # define (shared) gradient function

        self.loss_shared = loss_fnc(self.y_shared, self.t)
        self.gW_rec, self.gW_in, self.gW_out, \
            self.gb_rec, self.gb_out = TT.grad(self.loss_shared,
                                           [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
        self.grad_norm = TT.sqrt((self.gW_rec ** 2).sum() +
                                 (self.gW_in ** 2).sum() +
                                 (self.gW_out ** 2).sum() +
                                 (self.gb_rec ** 2).sum() +
                                 (self.gb_out ** 2).sum())

    def loss(self, W_rec, W_in, W_out, b_rec, b_out, u, t):
        y, _ = self.__net.net_output(W_rec, W_in, W_out, b_rec, b_out, u)
        return self.__loss_fnc(y, t)

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
