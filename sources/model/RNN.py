import os
import numpy
import theano as T
import theano.tensor as TT
from ActivationFunction import Tanh
from Configs import Configs
from Statistics import Statistics
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from initialization.GaussianInit import GaussianInit
from initialization.GivenValueInit import GivenValueInit
from initialization.ZeroInit import ZeroInit
from model.RnnVars import RnnVars

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

        # experimental
        self.experimental = RNN.Experimental(self)

        # build symbols
        self.__symbols = RNN.Symbols(self, W_rec, W_in, W_out, b_rec, b_out)

        # define visible functions
        self.net_output_shared = T.function([self.__symbols.u], self.__symbols.y_shared, name='net_output_shared_fun')

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
    def n_variables(self):
        return self.__n_out + self.__n_hidden + self.__n_hidden ** 2 \
               + self.__n_in * self.__n_hidden + self.__n_out * self.__n_hidden

    @property
    def symbols(self):
        return self.__symbols

    def from_tensor(self, v):
        n1 = self.__n_hidden ** 2
        n2 = n1 + self.__n_hidden * self.__n_in
        n3 = n2 + self.__n_hidden * self.__n_out
        n4 = n3 + self.__n_hidden
        n5 = n4 + self.__n_out

        W_rec_v = v[0:n1]
        W_in_v = v[n1:n2]
        W_out_v = v[n2:n3]
        b_rec_v = v[n3:n4]
        b_out_v = v[n4:n5]

        W_rec = TT.unbroadcast(W_rec_v.reshape((self.__n_hidden, self.__n_hidden), ndim=2), 0, 1)
        W_in = TT.unbroadcast(W_in_v.reshape((self.__n_hidden, self.__n_in), ndim=2), 0, 1)
        W_out = TT.unbroadcast(W_out_v.reshape((self.__n_out, self.__n_hidden), ndim=2), 0, 1)
        b_rec = TT.addbroadcast(TT.unbroadcast(b_rec_v.reshape((self.__n_hidden, 1)), 0), 1)
        b_out = TT.addbroadcast(TT.unbroadcast(b_out_v.reshape((self.__n_out, 1)), 0), 1)

        return RnnVars(self, W_rec, W_in, W_out, b_rec, b_out)

    #  should not be used from the outside
    def net_output(self, params: RnnVars, u):
        return self.__net_output(params.W_rec, params.W_in, params.W_out, params.b_rec, params.b_out, u)

    def __net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
        n_sequences = u.shape[2]
        h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.__n_hidden, n_sequences)

        values, _ = T.scan(self.net_output_t, sequences=u,
                           outputs_info=[h_m1, None, None],
                           non_sequences=[W_rec, W_in, W_out, b_rec, b_out],
                           name='net_output_scan')
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

    @property
    def spectral_radius(self):
        return numpy.max(abs(numpy.linalg.eigvals(self.symbols.current_params.W_rec.get_value())))

    @staticmethod
    def load_model(filename):
        npz = numpy.load(filename)

        W_rec = npz["W_rec"]
        W_in = npz["W_in"]
        W_out = npz["W_out"]
        b_rec = npz["b_rec"]
        b_out = npz["b_out"]

        n_hidden = npz["net_n_hidden"].item()
        n_in = npz["net_n_in"].item()
        n_out = npz["net_n_out"].item()

        activation_fnc = Tanh()  # FIXME XXX
        output_fnc = RNN.linear_fnc

        init_strategies = {'W_rec': GivenValueInit(W_rec), 'W_in': GivenValueInit(W_in), 'W_out': GivenValueInit(W_out),
                           'b_rec': GivenValueInit(b_rec), 'b_out': GivenValueInit(b_out)}

        return RNN(activation_fnc, output_fnc, n_hidden, n_in, n_out, init_strategies)

    @property
    def info(self):
        return InfoGroup('net',
                         InfoList(PrintableInfoElement('init_rho', ':2.2f', self.spectral_radius),
                                  PrintableInfoElement('n_hidden', ':d', self.__n_hidden),
                                  PrintableInfoElement('n_in', ':d', self.__n_in),
                                  PrintableInfoElement('n_out', ':d', self.__n_out),
                                  PrintableInfoElement('activation_fnc', '', self.__activation_fnc)
                                  ))

    def save_model(self, filename: str, stats: Statistics, train_info: Info):
        """saves the model with statistics to file"""

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        net_info_dict = self.info.dictionary
        stat_info_dict = stats.dictionary
        d = dict(W_rec=self.__symbols.current_params.W_rec.get_value(),
                 W_in=self.__symbols.current_params.W_in.get_value(),
                 W_out=self.__symbols.current_params.W_out.get_value(),
                 b_rec=self.__symbols.current_params.b_rec.get_value(),
                 b_out=self.__symbols.current_params.b_out.get_value())

        stat_info_dict.update(d)
        stat_info_dict.update(train_info.dictionary)
        stat_info_dict.update(net_info_dict)
        numpy.savez(filename + '.npz', **stat_info_dict)

    # predefined output functions
    @staticmethod
    def linear_fnc(y):
        return y

    @staticmethod
    def logistic(y):
        return 1. / (1. + TT.exp(-y))

    @staticmethod
    def softmax(y):
        return TT.nnet.softmax(y)

    # @staticmethod
    # def softmax(y):
    #     e_y = TT.exp(y - y.max(axis=0))
    #     return e_y / e_y.sum(axis=0)

    class Experimental:  # FIXME XXX
        def __init__(self, net):
            self.__net = net

        def net_output(self, params: RnnVars, u):
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
                               name='separate_matrices_net_output_scan',
                               n_steps=u.shape[0])
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
            self.__W_in = T.shared(W_in, name='W_in_shared')
            self.__W_rec = T.shared(W_rec, name='W_rec_shared')
            self.__W_out = T.shared(W_out, name='W_out_shared')
            self.__b_rec = T.shared(b_rec, name='b_rec_shared', broadcastable=(False, True))
            self.__b_out = T.shared(b_out, name='b_out_shared', broadcastable=(False, True))

            # current params
            self.__current_params = RnnVars(self.__net, self.__W_rec, self.__W_in, self.__W_out, self.__b_rec,
                                            self.__b_out)

            # define symbols
            W_in = TT.matrix(name='W_in')
            W_rec = TT.matrix(name='W_rec')
            W_out = TT.matrix(name='W_out')
            b_rec = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True), name='b_rec')
            b_out = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True), name='b_out')

            self.u = TT.tensor3(name='u')  # input tensor
            self.t = TT.tensor3(name='t')  # target tensor

            # output of the net
            self.y, self.deriv_a = net.net_output(self.__current_params, self.u)
            #self.y, self.deriv_a, *_ = net.experimental.net_output(self.__current_params, self.u)
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

        @property  # XXX
        def get_numeric_vector(self):
            return numpy.reshape(
                numpy.concatenate((self.__W_rec.get_value().flatten(), self.__W_in.get_value().flatten(),
                                   self.__W_out.get_value().flatten(), self.__b_rec.get_value().flatten(),
                                   self.__b_out.get_value().flatten())), (self.__net.n_variables, 1)).astype(dtype=Configs.floatType)
