import theano.tensor as TT
import theano as T
from theano.ifelse import ifelse

from infos.Info import Info
from infos.InfoList import InfoList
from infos.SymbolicInfo import SymbolicInfo
from model.RNNGradient import RNNGradient
from model.Variables import Variables
from theanoUtils import as_vector, norm2, normalize

__author__ = 'giulio'


class RNNVars(Variables):
    def __init__(self, net, W_rec, W_in, W_out, b_rec, b_out):
        self.__net = net
        self.__W_rec = W_rec
        self.__W_in = W_in
        self.__W_out = W_out
        self.__b_rec = b_rec
        self.__b_out = b_out

    def __as_tensor_list(self):
        return self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out

    def flatten(self):
        return as_vector(*self.__as_tensor_list())

    @property
    def net(self):
        return self.__net

    def as_tensor(self):
        return self.flatten()

    def dot(self, other):
        v1 = as_vector(*self.__as_tensor_list())
        v2 = as_vector(*other.__as_tensor_list())
        return TT.dot(v1.dimshuffle(1, 0), v2).squeeze()

    def cos(self, other):
        return self.dot(other) / (self.norm() * other.norm())

    def norm(self, L=2):  # XXX
        if L == 2:
            return norm2(self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out)  # FIXME
        elif L == 1:
            return (abs(self.flatten()).max())
        else:
            raise ValueError('unsupported norm {}'.format(L))

    def scale_norms_as(self, other):
        if not isinstance(other, RNNVars):
            raise TypeError('cannot perform this action with an object of type ' + str(type(other)))
        return RNNVars(self.__net, normalize(self.__W_rec) * other.W_rec.norm(2),
                       normalize(self.__W_in) * other.W_in.norm(2),
                       normalize(self.__W_out) * other.W_out.norm(2)
                       , normalize(self.__b_rec) * other.b_rec.norm(2),
                       normalize(self.__b_out) * other.b_out.norm(2))

    def __add__(self, other):
        if not isinstance(other, RNNVars):
            raise TypeError(
                'cannot add an object of type' + str(type(self)) + 'with an object of type ' + str(type(other)))
        return RNNVars(self.__net, self.__W_rec + other.__W_rec, self.__W_in + other.__W_in,
                       self.__W_out + other.__W_out,
                       self.__b_rec + other.__b_rec, self.__b_out + other.__b_out)

    def __sub__(self, other):
        if not isinstance(other, RNNVars):
            raise TypeError(
                'cannot subtract an object of type' + str(type(self)) + 'with an object of type ' + str(
                    type(other)))
        return RNNVars(self.__net, self.__W_rec - other.__W_rec, self.__W_in - other.__W_in,
                       self.__W_out - other.__W_out,
                       self.__b_rec - other.__b_rec, self.__b_out - other.__b_out)

    def __mul__(self, alpha):
        # if not isinstance(alpha, Number):
        #     raise TypeError('cannot multuple object of type ' + type(self),
        #                     ' with a non numeric type: ' + type(alpha))  # TODO theano scalar
        return RNNVars(self.__net, self.__W_rec * alpha, self.__W_in * alpha, self.__W_out * alpha,
                       self.__b_rec * alpha, self.__b_out * alpha)

    def __neg__(self):
        return self * (-1)

    @property
    def shape(self):
        return self.__net.n_variables, 1

    # XXX mettere a pulito
    def step_as_direction(self, strategy):

        lr_w_rec, lr_w_rec_symbolic_infos = strategy.compute_lr(self.__net, None, self.__W_rec)
        lr_w_in, lr_w_in_symbolic_infos = strategy.compute_lr(self.__net, None, self.__W_in)
        lr_w_out, lr_w_out_symbolic_infos = strategy.compute_lr(self.__net, None, self.__W_out)
        lr_b_rec, lr_b_rec_symbolic_infos = strategy.compute_lr(self.__net, None, self.__b_rec)
        lr_b_out, lr_b_out_symbolic_infos = strategy.compute_lr(self.__net, None, self.__b_out)

        #lr_w_out *= 0.5
        #lr_b_out *= 0.5

        #lr_w_rec *= 0.5
        #lr_b_rec *= 0.5

        info = RNNVars.StepInfo(lr_w_rec_symbolic_infos, lr_w_in_symbolic_infos, lr_w_out_symbolic_infos,
                                lr_b_rec_symbolic_infos, lr_b_out_symbolic_infos)

        return RNNVars(self.__net, self.__W_rec * lr_w_rec, self.__W_in * lr_w_in, self.__W_out * lr_w_out,
                       self.__b_rec * lr_b_rec,
                       self.__b_out * lr_b_out), info

    class StepInfo(SymbolicInfo):

        def __init__(self, lr_w_rec_symbolic_infos, lr_w_in_symbolic_infos, lr_w_out_symbolic_infos,
                     lr_b_rec_symbolic_infos, lr_b_out_symbolic_infos):
            self.__lr_w_rec_symbolic_infos = lr_w_rec_symbolic_infos
            self.__lr_w_in_symbolic_infos = lr_w_in_symbolic_infos
            self.__lr_w_out_symbolic_infos = lr_w_out_symbolic_infos
            self.__lr_b_rec_symbolic_infos = lr_b_rec_symbolic_infos
            self.__lr_b_out_symbolic_infos = lr_b_out_symbolic_infos

            self.__symbols = lr_w_rec_symbolic_infos.symbols + lr_w_in_symbolic_infos.symbols + \
                             lr_w_out_symbolic_infos.symbols + lr_b_rec_symbolic_infos.symbols + \
                             lr_b_out_symbolic_infos.symbols

        @property
        def symbols(self):
            return self.__symbols

        def fill_symbols(self, symbols_replacedments: list) -> Info:
            lr_w_rec_info = self.__lr_w_rec_symbolic_infos.fill_symbols(symbols_replacedments)
            i = len(self.__lr_w_rec_symbolic_infos.symbols)
            lr_w_in_info = self.__lr_w_in_symbolic_infos.fill_symbols(
                symbols_replacedments[i:])
            i += len(self.__lr_w_in_symbolic_infos.symbols)
            lr_w_out_info = self.__lr_w_out_symbolic_infos.fill_symbols(
                symbols_replacedments[i:])
            i += len(self.__lr_w_out_symbolic_infos.symbols)
            lr_b_rec_info = self.__lr_b_rec_symbolic_infos.fill_symbols(
                symbols_replacedments[i:])
            i += len(self.__lr_b_rec_symbolic_infos.symbols)
            lr_b_out_info = self.__lr_b_out_symbolic_infos.fill_symbols(
                symbols_replacedments[i:])
            return InfoList(lr_w_rec_info, lr_w_in_info, lr_w_out_info, lr_b_rec_info, lr_b_out_info)

    # XXX
    def to_zero_if(self, condition):
        self.__W_rec = ifelse(condition, TT.zeros_like(self.__W_rec), self.__W_rec)
        self.__W_in = ifelse(condition, TT.zeros_like(self.__W_in), self.__W_in)
        self.__W_out = ifelse(condition, TT.zeros_like(self.__W_out), self.__W_out)
        self.__b_rec = ifelse(condition, TT.zeros_like(self.__b_rec), self.__b_rec)
        self.__b_out = ifelse(condition, TT.zeros_like(self.__b_out), self.__b_out)

    def net_output(self, u):
        return self.__net.net_output(self, u)

    # TODO REMOVE
    def setW_rec(self, W_rec):
        self.__W_rec = W_rec

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

    @property
    def net(self):
        return self.__net

    def update_list(self, other):
        return [
            (self.__W_rec, other.W_rec),
            (self.__W_in, other.W_in),
            (self.__W_out, other.W_out),
            (self.__b_rec, TT.addbroadcast(other.b_rec, 1)),
            (self.__b_out, TT.addbroadcast(other.b_out, 1))]
