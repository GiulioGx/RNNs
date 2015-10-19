from Penalty import Penalty, NullPenalty
import theano.tensor as TT
import theano as T
from theanoUtils import norm

__author__ = 'giulio'


class ObjectiveFunction(object):
    def __init__(self, loss_fnc, penalty: Penalty = NullPenalty()):
        self.__loss_fnc = loss_fnc
        self.__penalty = penalty

    def obj_symbols(self, net):
        return ObjectiveSymbolCloset(net, self.__loss_fnc, self.__penalty)

    def loss(self, y, t):
        return self.__loss_fnc(y, t)


class ObjectiveSymbolCloset(object):
    def __init__(self, net, loss_fnc, penalty: Penalty):
        self.__net = net
        self.__loss_fnc = loss_fnc
        self.__penalty = penalty

        # define symbols
        self.__loss_shared = self.__loss_fnc(net.symb_closet.y_shared, net.symb_closet.t)
        self.gW_rec, self.gW_in, self.gW_out, \
        self.gb_rec, self.gb_out = TT.grad(self.__loss_shared,
                                           [net.symb_closet.W_rec, net.symb_closet.W_in, net.symb_closet.W_out,
                                            net.symb_closet.b_rec, net.symb_closet.b_out])
        self.grad_norm = norm(self.gW_rec, self.gW_in, self.gW_out, self.gb_rec, self.gb_out)

        self.net_loss = T.function([net.symb_closet.u, net.symb_closet.t], self.__loss_shared)

    def loss(self, W_rec, W_in, W_out, b_rec, b_out, u, t):
        y, _ = self.__net.net_output(W_rec, W_in, W_out, b_rec, b_out, u)
        return self.__loss_fnc(y, t)



        # TODO add penalty
