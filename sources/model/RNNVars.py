import theano.tensor as TT
from model.RNNGradient import RnnGradient
from model.Variables import Variables
from theanoUtils import norm, as_vector, norm2

__author__ = 'giulio'


class RnnVars(Variables):
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
        # v1 =  self.__W_rec.flatten().dimshuffle(0, 'x')
        # v2 =  self.__W_in.flatten().dimshuffle(0, 'x')
        # v3 =  self.__W_out.flatten().dimshuffle(0, 'x')
        # v4 =  self.__b_rec.flatten().dimshuffle(0, 'x')
        # v5 =  self.__b_out.flatten().dimshuffle(0, 'x')
        #
        # return TT.concatenate([v1, v2, v3, v4, v5])
        return self.flatten()

    def dot(self, other):
        v1 = as_vector(*self.__as_tensor_list())
        v2 = as_vector(*other.__as_tensor_list())
        return TT.dot(v1.dimshuffle(1, 0), v2).squeeze()

    def cos(self, other):
        return self.dot(other) / (self.norm() * other.norm())

    def norm(self):
        return norm2(self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out)

    def scale_norms_as(self, other):
        if not isinstance(other, RnnVars):
            raise TypeError('cannot perform this action with an object of type ' + type(other))
        return RnnVars(self.__net, self.__W_rec / norm(self.__W_rec) * norm(other.W_rec),
                       self.__W_in / norm(self.__W_in) * norm(other.W_in),
                       self.__W_out / norm(self.__W_out) * norm(other.W_out)
                       , self.__b_rec / norm(self.__b_rec) * norm(other.b_rec),
                       self.__b_out / norm(self.__b_out) * norm(other.b_out))

    def __add__(self, other):
        if not isinstance(other, RnnVars):
            raise TypeError('cannot add an object of type' + type(self) + 'with an object of type ' + type(other))
        return RnnVars(self.__net, self.__W_rec + other.__W_rec, self.__W_in + other.__W_in,
                       self.__W_out + other.__W_out,
                       self.__b_rec + other.__b_rec, self.__b_out + other.__b_out)

    def __sub__(self, other):
        if not isinstance(other, RnnVars):
            raise TypeError(
                'cannot subtract an object of type' + type(self) + 'with an object of type ' + type(other))
        return RnnVars(self.__net, self.__W_rec - other.__W_rec, self.__W_in - other.__W_in,
                       self.__W_out - other.__W_out,
                       self.__b_rec - other.__b_rec, self.__b_out - other.__b_out)

    def __mul__(self, alpha):
        # if not isinstance(alpha, Number):
        #     raise TypeError('cannot multuple object of type ' + type(self),
        #                     ' with a non numeric type: ' + type(alpha))  # TODO theano scalar
        return RnnVars(self.__net, self.__W_rec * alpha, self.__W_in * alpha, self.__W_out * alpha,
                       self.__b_rec * alpha, self.__b_out * alpha)

    def __neg__(self):
        return self * (-1)

    def gradient(self, loss_fnc, u, t):
        return RnnGradient(self, loss_fnc, u, t)

    def failsafe_grad(self, loss_fnc, u, t):  # FIXME XXX remove me
        y, _, _ = self.__net.net_output(self, u, self.__net.symbols.h_m1)
        loss = loss_fnc.value(y, t)
        gW_rec, gW_in, gW_out, \
        gb_rec, gb_out = TT.grad(loss, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
        return RnnVars(self.__net, gW_rec, gW_in, gW_out, gb_rec, gb_out)

    # XXX
    def to_zero_if(self, condition):
        self.__W_rec = TT.switch(condition, TT.zeros_like(self.__W_rec), self.__W_rec)
        self.__W_in = TT.switch(condition, TT.zeros_like(self.__W_in), self.__W_in)
        self.__W_out = TT.switch(condition, TT.zeros_like(self.__W_out), self.__W_out)
        self.__b_rec = TT.switch(condition, TT.zeros_like(self.__b_rec), self.__b_rec)
        self.__b_out= TT.switch(condition, TT.zeros_like(self.__b_out), self.__b_out)


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
