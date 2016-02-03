import theano.tensor as TT
from theano.ifelse import ifelse

from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.InfoList import InfoList
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
        return norm2(self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out)

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

    # XXX mettere a pulito
    def format_lr_info(self, infos_symbols):
        lr_w_rec_info, infos_symbols = self.__lr_w_rec_symbols.format_infos(infos_symbols)
        #print(str(lr_w_rec_info))
        lr_w_in_info, infos_symbols = self.__lr_w_in_symbols.format_infos(infos_symbols)
        lr_w_out_info, infos_symbols = self.__lr_w_out_symbols.format_infos(infos_symbols)
        lr_b_rec_info, infos_symbols = self.__lr_b_rec_symbols.format_infos(infos_symbols)
        lr_b_out_info, infos_symbols = self.__lr_b_out_symbols.format_infos(infos_symbols)
        return InfoList(lr_w_rec_info, lr_w_in_info, lr_w_out_info, lr_b_rec_info, lr_b_out_info), infos_symbols
        #return lr_w_rec_info, infos_symbols

    # XXX mettere a pulito
    def step_as_direction(self, strategy):

        class VarsSymbol(DescentDirectionRule.Symbols):
            def __init__(self, vars):
                self.__vars = vars

            @property
            def direction(self):
                return self.__vars

            def format_infos(self, infos):
                pass

            def infos(self):
                pass

        self.__lr_w_rec_symbols = strategy.compile(self.__net, None, VarsSymbol(self.__W_rec))
        lr_w_rec = self.__lr_w_rec_symbols.learning_rate
        lr_w_rec_infos = self.__lr_w_rec_symbols.infos

        self.__lr_w_in_symbols = strategy.compile(self.__net, None, VarsSymbol(self.__W_in))
        lr_w_in = self.__lr_w_in_symbols.learning_rate
        lr_w_in_infos = self.__lr_w_in_symbols.infos

        self.__lr_w_out_symbols = strategy.compile(self.__net, None, VarsSymbol(self.__W_out))
        lr_w_out = self.__lr_w_out_symbols.learning_rate
        lr_w_out_infos = self.__lr_w_out_symbols.infos

        self.__lr_b_rec_symbols = strategy.compile(self.__net, None, VarsSymbol(self.__b_rec))
        lr_b_rec = self.__lr_b_rec_symbols.learning_rate
        lr_b_rec_infos = self.__lr_b_rec_symbols.infos

        self.__lr_b_out_symbols = strategy.compile(self.__net, None, VarsSymbol(self.__b_out))
        lr_b_out = self.__lr_b_out_symbols.learning_rate
        lr_b_out_infos = self.__lr_b_out_symbols.infos

        return RNNVars(self.__net, self.__W_rec * lr_w_rec, self.__W_in * lr_w_in, self.__W_out * lr_w_out,
                       self.__b_rec * lr_b_rec,
                       self.__b_out * lr_b_out), lr_w_rec_infos + lr_w_in_infos + lr_w_out_infos + lr_b_rec_infos + lr_b_out_infos

    def gradient(self, loss_fnc, u, t):
        return RNNGradient(self, loss_fnc, u, t)

    def failsafe_grad(self, loss_fnc, u, t):  # FIXME XXX remove me
        y, _, _ = self.__net.net_output(self, u, self.__net.symbols.h_m1)
        loss = loss_fnc.value(y, t)
        gW_rec, gW_in, gW_out, \
        gb_rec, gb_out = TT.grad(loss, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
        return RNNVars(self.__net, W_rec=gW_rec, W_in=gW_in, W_out=gW_out, b_rec=gb_rec, b_out=gb_out)

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
