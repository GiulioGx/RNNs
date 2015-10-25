from numbers import Number
import theano as T
import theano.tensor as TT
import numpy
import os
from Configs import Configs
import theano.typed_list
from Params import Params
from Statistics import Statistics
from theanoUtils import norm

__author__ = 'giulio'


class RNN(object):
    def __init__(self, activation_fnc, output_fnc, n_hidden, n_in, n_out, seed):
        # topology
        self.__n_hidden = n_hidden
        self.__n_in = n_in
        self.__n_out = n_out

        # activation functions
        self.__activation_fnc = activation_fnc

        # output function
        self.__output_fnc = output_fnc

        # random generator
        self.__rng = numpy.random.RandomState(seed)

        # init weight matrices TODO
        scale = .14
        loc = 0
        W_in = numpy.asarray(
            self.__rng.normal(size=(self.__n_hidden, self.__n_in), scale=scale, loc=loc), dtype=Configs.floatType)
        W_rec = numpy.asarray(
            self.__rng.normal(size=(self.__n_hidden, self.__n_hidden), scale=scale, loc=loc), dtype=Configs.floatType)
        W_out = numpy.asarray(
            self.__rng.normal(size=(self.__n_out, self.__n_hidden), scale=scale, loc=loc), dtype=Configs.floatType)

        # init biases
        b_rec = numpy.zeros((self.__n_hidden, 1), Configs.floatType)
        b_out = numpy.zeros((self.__n_out, 1), Configs.floatType)

        # build symbol closet
        self.__symbols = RNN.Symbols(self, W_rec, W_in, W_out, b_rec, b_out)

        # experimental
        self.experimental = RNN.Experimental(self)

        # define visible functions
        self.net_output_shared = T.function([self.__symbols.u], self.__symbols.y_shared)

    @property
    def n_hidden(self):
        return self.__n_hidden

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def symbols(self):
        return self.__symbols

    def net_output(self, params, u):
        return self.__net_output(params.W_rec, params.W_in, params.W_out, params.b_rec, params.b_out, u)

    def __net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
        n_sequences = u.shape[2]
        h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.__n_hidden, n_sequences)

        values, _ = T.scan(self.net_output_t, sequences=u,
                           outputs_info=[h_m1, None, None],
                           non_sequences=[W_rec, W_in, W_out, b_rec, b_out],
                           name='net_output',
                           mode=T.Mode(linker='cvm'))
        y = values[1]
        deriv_a = values[2]
        return y, deriv_a

    def net_output_t(self, u_t, h_tm1, W_rec, W_in, W_out, b_rec, b_out):
        h_t, deriv_a_t = self.h_t(u_t, h_tm1, W_rec, W_in, b_rec)
        y_t = self.y_t(h_t, W_out, b_out)
        return h_t, y_t, deriv_a_t

    def h_t(self, u_t, h_tm1, W_rec, W_in, b_rec):
        a_t = TT.dot(W_rec, h_tm1) + TT.dot(W_in, u_t) + b_rec
        deriv_a = self.__activation_fnc.grad_f(a_t)
        return self.__activation_fnc.f(a_t), deriv_a

    # single output mode
    # def __y(self, h):
    #     return self.__output_fnc(TT.dot(self.__W_out, h[-1]) + self.__b_out)

    def y_t(self, h_t, W_out, b_out):
        return self.__output_fnc(TT.dot(W_out, h_t) + b_out)

    def save_model(self, path, filename, stats: Statistics):
        """saves the model with statistics to file"""

        os.makedirs(path, exist_ok=True)

        info_dict = stats.dictionary
        d = dict(n_hidden=self.__n_hidden,
                 n_in=self.__n_in,
                 n_out=self.__n_out,
                 activation_fnc=str(self.__activation_fnc),
                 W_rec=self.__symbols.current_params.W_rec.get_value(),
                 W_in=self.__symbols.current_params.W_in.get_value(),
                 W_out=self.__symbols.current_params.W_out.get_value(),
                 b_rec=self.__symbols.current_params.b_rec.get_value(),
                 b_out=self.__symbols.current_params.b_out.get_value())

        info_dict.update(d)
        numpy.savez(path + '/' + filename + '.npz', **info_dict)

    # predefined output functions
    @staticmethod
    def last_linear_fnc(y):
        return y

    class Params(Params):

        def __init__(self, W_rec, W_in, W_out, b_rec, b_out):
            self.__W_rec = W_rec
            self.__W_in = W_in
            self.__W_out = W_out
            self.__b_rec = b_rec
            self.__b_out = b_out

        def __add__(self, other):
            if not isinstance(other, RNN.Params):
                raise TypeError('cannot add an object of type' + type(self) + 'with an object of type ' + type(other))
            return RNN.Params(self.__W_rec + other.__W_rec, self.__W_in + other.__W_in, self.__W_out + other.__W_out,
                              self.__b_rec + other.__b_rec, self.__b_out + other.__b_out)

        def __mul__(self, alpha):
            # if not isinstance(alpha, Number):
            #     raise TypeError('cannot multuple object of type ' + type(self),
            #                     ' with a non numeric type: ' + type(alpha))  # TODO theano scalar
            return RNN.Params(self.__W_rec * alpha, self.__W_in * alpha, self.__W_out * alpha,
                              self.__b_rec * alpha, self.__b_out * alpha)

        def norm(self):
            return norm(self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out)

        def grad(self, fnc):
            gW_rec, gW_in, gW_out, \
            gb_rec, gb_out = TT.grad(fnc, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
            return RNN.Params(gW_rec, gW_in, gW_out, gb_rec, gb_out)

        def update_dictionary(self, other):
            return [
                (self.__W_rec, other.W_rec),
                (self.__W_in, other.W_in),
                (self.__W_out, other.W_out),
                (self.__b_rec, other.b_rec),
                (self.__b_out, other.b_out)]

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

    class Experimental:
        def __init__(self, net):
            self.__net = net

        def net_output(self, params, u):
            return self.__net_output(params.W_rec, params.W_in, params.W_out, params.b_rec, params.b_out, u)

        def __net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
            W_fixes = []
            for i in range(200):
                W_fixes.append(W_rec.clone())

            n_sequences = u.shape[2]
            h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.__net.n_hidden, n_sequences)

            values, _ = T.scan(self.__net_output_t, sequences=[u, TT.as_tensor_variable(W_fixes)],
                               outputs_info=[h_m1, None, None],
                               non_sequences=[W_in, W_out, b_rec, b_out],
                               name='net_output',
                               mode=T.Mode(linker='cvm'),
                               n_steps=u.shape[0])
            y = values[1]
            deriv_a = values[2]
            return y, deriv_a, W_fixes

        def __net_output_t(self, u_t, W_rec, h_tm1, W_in, W_out, b_rec, b_out):
            h_t, deriv_a_t = self.__net.h_t(u_t, h_tm1, W_rec, W_in, b_rec)
            y_t = self.__net.y_t(h_t, W_out, b_out)
            return h_t, y_t, deriv_a_t

    class Symbols:
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

            self.__current_params = RNN.Params(self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out)

            self.u = TT.tensor3()  # input tensor
            self.t = TT.tensor3()  # target tensor

            self.y, self.deriv_a = net.net_output(self.__current_params, self.u)
            self.y_shared, self.deriv_a_shared = T.clone([self.y, self.deriv_a],
                                                         replace={W_rec: self.__W_rec, W_in: self.__W_in,
                                                                  W_out: self.__W_out, b_rec: self.__b_rec,
                                                                  b_out: self.__b_out})

        def get_deriv_a(self, params):
            _, deriv_a = self.__net.net_output(params, self.u)
            return deriv_a

        @property
        def current_params(self):
            return self.__current_params
