import theano as T
import theano.tensor as TT
import numpy
import os
from Configs import Configs
import theano.typed_list

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

    def net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
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

    def save_model(self, path, filename, stats):
        """saves the model with statistics to file"""

        os.makedirs(path, exist_ok=True)

        numpy.savez(path + '/' + filename + '.npz',
                    n_hidden=self.__n_hidden,
                    n_in=self.__n_in,
                    n_out=self.__n_out,
                    activation_fnc=str(self.__activation_fnc),
                    valid_error=stats.valid_error,
                    gradient_norm=stats.grad_norm,
                    rho=stats.rho,
                    penalty=stats.penalty_norm,
                    elapsed_time=stats.elapsed_time,
                    W_rec=self.__symbols.W_rec.get_value(),
                    W_in=self.__symbols.W_in.get_value(),
                    W_out=self.__symbols.W_out.get_value(),
                    b_rec=self.__symbols.b_rec.get_value(),
                    b_out=self.__symbols.b_out.get_value())

    # predefined output functions
    @staticmethod
    def last_linear_fnc(y):
        return y

    class Experimental:
        def __init__(self, net):
            self.__net = net

        def net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):

                W_fixes = []
                for i in range(200):
                    W_fixes.append(self.__net.symbols.W_rec.clone())

                n_sequences = u.shape[2]
                h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.__net.n_hidden, n_sequences)

                values, _ = T.scan(self.__net_output_t, sequences=[u, TT.as_tensor_variable(W_fixes)],
                                   outputs_info=[h_m1, None, None],
                                   non_sequences=[W_in, W_out, b_rec, b_out],
                                   name='net_output',
                                   mode=T.Mode(linker='cvm'))
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
