import os
import theano as T
import theano.tensor as TT
import numpy
from ActivationFunction import Relu
from Configs import Configs
from InfoProducer import InfoProducer
from combiningRule.CombiningRule import CombiningRule
from infos.Info import Info
from Params import Params
from Statistics import Statistics
from infos.InfoElement import NonPrintableInfoElement
from initialization.GaussianInit import GaussianInit
from initialization.GivenValueInit import GivenValueInit
from initialization.ZeroInit import ZeroInit
from theanoUtils import norm, as_vector

__author__ = 'giulio'


class RNN(object):
    deafult_init_strategies = {'W_rec': GaussianInit(), 'W_in': GaussianInit(), 'W_out': GaussianInit(),
                               'b_rec': ZeroInit(), 'b_out': ZeroInit()}

    def __init__(self, activation_fnc, output_fnc, n_hidden, n_in, n_out, init_strategies: deafult_init_strategies,
                 seed=Configs.seed):
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

        W_rec = init_strategies['W_rec'].init_matrix((self.__n_hidden, self.__n_hidden), Configs.floatType)
        W_in = init_strategies['W_in'].init_matrix((self.__n_hidden, self.__n_in), Configs.floatType)
        W_out = init_strategies['W_out'].init_matrix((self.__n_out, self.__n_hidden), Configs.floatType)

        # init biases
        b_rec = init_strategies['b_rec'].init_matrix((self.__n_hidden, 1), Configs.floatType)
        b_out = init_strategies['b_out'].init_matrix((self.__n_out, 1), Configs.floatType)

        # build symbols
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

    @staticmethod
    def load_model(filename):
        npz = numpy.load(filename)

        W_rec = npz["W_rec"]
        W_in = npz["W_in"]
        W_out = npz["W_out"]
        b_rec = npz["b_rec"]
        b_out = npz["b_out"]

        n_hidden = npz["n_hidden"]
        n_in = npz["n_in"]
        n_out = npz["n_out"]

        activation_fnc = Relu()  # FIXME XXX
        output_fnc = RNN.linear_fnc

        init_strategies = {'W_rec': GivenValueInit(W_rec), 'W_in': GivenValueInit(W_in), 'W_out': GivenValueInit(W_out),
                           'b_rec': GivenValueInit(b_rec), 'b_out': GivenValueInit(b_out)}

        return RNN(activation_fnc, output_fnc, n_hidden, n_in, n_out, init_strategies)

    def save_model(self, filename, stats: Statistics, info: Info):
        """saves the model with statistics to file"""

        os.makedirs(os.path.dirname(filename), exist_ok=True)

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
        info_dict.update(info.dictionary)
        numpy.savez(filename + '.npz', **info_dict)

    # predefined output functions
    @staticmethod
    def linear_fnc(y):
        return y

    # @staticmethod
    # def softmax(y):
    #     e_y = TT.exp(y - y.max(axis=0))
    #     return e_y / e_y.sum(axis=0)

    class Params(Params):

        def __as_tensor_list(self):
            return self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out

        def flatten(self):
            return as_vector(self.__as_tensor_list())

        @staticmethod
        def from_flattened_tensor(v, net):

            n1 = net.n_hidden ** 2
            n2 = n1 + net.n_hidden * net.n_in
            n3 = n2 + net.n_hidden * net.n_out
            n4 = n3 + net.n_hidden
            n5 = n4 + net.n_out

            W_rec_v = v[0:n1]
            W_in_v = v[n1:n2]
            W_out_v = v[n2:n3]
            b_rec_v = v[n3:n4]
            b_out_v = v[n4:n5]

            W_rec = W_rec_v.reshape((net.n_hidden, net.n_hidden))
            W_in = W_in_v.reshape((net.n_hidden, net.n_in))
            W_out = W_out_v.reshape((net.n_out, net.n_hidden))
            b_rec = b_rec_v.reshape((net.n_hidden, 1))
            b_out = b_out_v.reshape((net.n_out, 1))

            return RNN.Params(net, W_rec, W_in, W_out, b_rec, b_out)

        def dot(self, other):
            v1 = as_vector(*self.__as_tensor_list())
            v2 = as_vector(*other.__as_tensor_list())
            return TT.dot(v1.dimshuffle(1, 0), v2)

        def __init__(self, net, W_rec, W_in, W_out, b_rec, b_out):
            self.__net = net
            self.__W_rec = W_rec
            self.__W_in = W_in
            self.__W_out = W_out
            self.__b_rec = b_rec
            self.__b_out = b_out

        def __add__(self, other):
            if not isinstance(other, RNN.Params):
                raise TypeError('cannot add an object of type' + type(self) + 'with an object of type ' + type(other))
            return RNN.Params(self.__net, self.__W_rec + other.__W_rec, self.__W_in + other.__W_in,
                              self.__W_out + other.__W_out,
                              self.__b_rec + other.__b_rec, self.__b_out + other.__b_out)

        def __sub__(self, other):
            if not isinstance(other, RNN.Params):
                raise TypeError(
                    'cannot subtract an object of type' + type(self) + 'with an object of type ' + type(other))
            return RNN.Params(self.__net, self.__W_rec - other.__W_rec, self.__W_in - other.__W_in,
                              self.__W_out - other.__W_out,
                              self.__b_rec - other.__b_rec, self.__b_out - other.__b_out)

        def __mul__(self, alpha):
            # if not isinstance(alpha, Number):
            #     raise TypeError('cannot multuple object of type ' + type(self),
            #                     ' with a non numeric type: ' + type(alpha))  # TODO theano scalar
            return RNN.Params(self.__net, self.__W_rec * alpha, self.__W_in * alpha, self.__W_out * alpha,
                              self.__b_rec * alpha, self.__b_out * alpha)

        def norm(self):
            return norm(self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out)

        def grad(self, fnc):
            gW_rec, gW_in, gW_out, \
            gb_rec, gb_out = TT.grad(fnc, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
            return RNN.Params(self.__net, gW_rec, gW_in, gW_out, gb_rec, gb_out)

        def grad_combining_steps(self, loss_fnc, strategy: CombiningRule, u, t):
            return RNN.Params.SeparateGrads(self, loss_fnc, strategy, u, t)

        class SeparateGrads(InfoProducer):

            @property
            def infos(self):
                return self.__info

            def format_infos(self, info_symbols):
                separate_norms_dict = {'W_rec': info_symbols[0], 'W_in': info_symbols[1], 'W_out': info_symbols[2],
                                       'b_rec': info_symbols[3],
                                       'b_out': info_symbols[4]}
                info = NonPrintableInfoElement('separate_norms', separate_norms_dict)
                return info, info_symbols[5: len(info_symbols)]

            @property
            def grad_combination(self):
                return self.__grad_combination

            def __init__(self, params, loss_fnc, strategy: CombiningRule, u, t):
                y, _, W_rec_fixes, W_in_fixes, W_out_fixes, b_rec_fixes, b_out_fixes = params.net.experimental.net_output(
                    params, u)
                loss = loss_fnc(y, t)

                gW_rec_list = T.grad(loss, W_rec_fixes)
                gW_in_list = T.grad(loss, W_in_fixes)
                gW_out_list = T.grad(loss, W_out_fixes)
                gb_rec_list = T.grad(loss, b_rec_fixes)
                gb_out_list = T.grad(loss, b_out_fixes)

                l = u.shape[0]

                self.__grad_combination, gW_rec_norms, gW_in_norms, gW_out_norms, \
                gb_rec_norms, gb_out_norms = self.__get_combination_togheter(
                    gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, l, strategy, params.net)

                self.__info = [gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms]

            def __get_combination_separately(self, gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, l, strategy, net):
                    gW_rec_combinantion, gW_rec_norms = strategy.combine(gW_rec_list, l)
                    gW_in_combinantion, gW_in_norms = strategy.combine(gW_in_list, l)
                    gW_out_combinantion, gW_out_norms = strategy.combine(gW_out_list, l)
                    gb_rec_combinantion, gb_rec_norms = strategy.combine(gb_rec_list, l)
                    gb_out_combinantion, gb_out_norms = strategy.combine(gb_out_list, l)

                    combination = RNN.Params(net, gW_rec_combinantion, gW_in_combinantion,
                                                     gW_out_combinantion,
                                                     gb_rec_combinantion, gb_out_combinantion)

                    return combination, gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms

            def get_norms(self, list, n):

                values, _ = T.scan(norm, sequences=[TT.as_tensor_variable(list)],
                           outputs_info=[None],
                           non_sequences=[],
                           name='get_norms',
                           mode=T.Mode(linker='cvm'),
                           n_steps=n)

                return values

            def __get_combination_togheter(self, gW_rec_list, gW_in_list, gW_out_list, gb_rec_list, gb_out_list, l, strategy, net):

                    gW_rec_norms = self.get_norms(gW_rec_list, l)
                    gW_in_norms = self.get_norms(gW_in_list, l)
                    gW_out_norms = self.get_norms(gW_out_list, l)
                    gb_rec_norms = self.get_norms(gb_rec_list, l)
                    gb_out_norms = self.get_norms(gb_out_list, l)

                    values, _ = T.scan(as_vector, sequences=[TT.as_tensor_variable(gW_rec_list), TT.as_tensor_variable(gW_in_list),
                                                        TT.as_tensor_variable(gW_out_list), TT.as_tensor_variable(gb_rec_list),
                                                        TT.as_tensor_variable(gb_out_list)],
                           outputs_info=[None],
                           non_sequences=[],
                           name='as_vector_combinations',
                           mode=T.Mode(linker='cvm'),
                           n_steps=l)

                    combination_v, norms = strategy.combine(values, l)
                    combination = RNN.Params.from_flattened_tensor(combination_v, net)

                    return combination, gW_rec_norms, gW_in_norms, gW_out_norms, gb_rec_norms, gb_out_norms

        def update_dictionary(self, other):
            return [
                (self.__W_rec, other.W_rec),
                (self.__W_in, other.W_in),
                (self.__W_out, other.W_out),
                (self.__b_rec, TT.addbroadcast(other.b_rec, 1)),
                (self.__b_out, TT.addbroadcast(other.b_out, 1))]

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

    class Experimental:  # FIXME XXX
        def __init__(self, net):
            self.__net = net

        def net_output(self, params, u):
            return self.__net_output(params.W_rec, params.W_in, params.W_out, params.b_rec, params.b_out, u)

        def __net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
            W_rec_fixes = []
            W_in_fixes = []
            W_out_fixes = []
            b_rec_fixes = []
            b_out_fixes = []

            for i in range(200):  # FIXME max_lenght
                W_rec_fixes.append(W_rec.clone())
                W_in_fixes.append(W_in.clone())
                W_out_fixes.append(W_out.clone())
                b_rec_fixes.append(b_rec.clone())
                b_out_fixes.append(b_out.clone())

            n_sequences = u.shape[2]
            h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.__net.n_hidden, n_sequences)

            values, _ = T.scan(self.__net_output_t,
                               sequences=[u, TT.as_tensor_variable(W_rec_fixes), TT.as_tensor_variable(W_in_fixes),
                                          TT.as_tensor_variable(W_out_fixes), TT.as_tensor_variable(b_rec_fixes),
                                          TT.as_tensor_variable(b_out_fixes)],
                               outputs_info=[h_m1, None, None],
                               non_sequences=[],
                               name='net_output',
                               mode=T.Mode(linker='cvm'),
                               n_steps=u.shape[0])  # TODO check if +1
            y = values[1]
            deriv_a = values[2]
            return y, deriv_a, W_rec_fixes, W_in_fixes, W_out_fixes, b_rec_fixes, b_out_fixes

        def __net_output_t(self, u_t, W_rec_fixes, W_in_fixes, W_out_fixes, b_rec_fixes, b_out_fixes, h_tm1):
            h_t, deriv_a_t = self.__net.h_t(u_t, h_tm1, W_rec_fixes, W_in_fixes, b_rec_fixes)
            y_t = self.__net.y_t(h_t, W_out_fixes, b_out_fixes)
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

            # current params
            self.__current_params = RNN.Params(self.__net, self.__W_rec, self.__W_in, self.__W_out, self.__b_rec,
                                               self.__b_out)

            # define symbols
            W_in = TT.matrix()
            W_rec = TT.matrix()
            W_out = TT.matrix()
            b_rec = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True))
            b_out = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True))

            self.u = TT.tensor3()  # input tensor
            self.t = TT.tensor3()  # target tensor

            # output of the net
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
