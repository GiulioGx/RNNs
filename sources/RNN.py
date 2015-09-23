import theano as T
import theano.tensor as TT
import numpy
from configs import Configs

__author__ = 'giulio'


class RNN:
    def __init__(self, task, activation_fnc, output_fnc, n_hidden):

        # topology
        self.__n_hidden = n_hidden
        self.__n_in = task.n_in
        self.__n_out = task.n_out

        # activation fnc
        self.__activation_fnc = activation_fnc

        #output_fnc
        self.__output_fnc = output_fnc


        # task
        self.__task = task

        # init weight matrices
        W_in = numpy.asarray(
            self.__rng.normal(size=(self.__n_hidden, self.__nin), scale=.01, loc=.0), dtype=Configs.floatType)
        W_rec = numpy.asarray(
            self.__rng.normal(size=(self.__n_hidden, self.__n_hidden), scale=.01, loc=.0), dtype=Configs.floatType)
        W_out = numpy.asarray(
            self.__rng.normal(size=(self.__nout, self.__n_hidden), scale=.01, loc=0.0), dtype=Configs.floatType)

        # init biases
        b_rec = numpy.zeros((self.__n_hidden, 1), Configs.floatType)
        b_out = numpy.zeros((self.__nout, 1), Configs.floatType)

        self.__W_in = T.shared(W_in, 'W_in')
        self.__W_rec = T.shared(W_rec, 'W_rec')
        self.__W_out = T.shared(W_out, 'W_out')
        self.__b_rec = T.shared(b_rec, 'b_rec')
        self.__b_out = T.shared(b_out, 'b_out')

    def __h(self, h0_tm1, u):

        def h_t(u_t, h_tm1):
            return self.__activation_fnc(TT.dot(self.__W_rec, h_tm1) + TT.dot(self.__W_in, u_t) + self.__b_rec)

        h, _ = T.scan(h_t, sequences=u,
                      outputs_info=[h0_tm1],
                      non_sequences=[self.__W_rec, self.__W_in],
                      name='h_t',
                      mode=T.Mode(linker='cvm'))
        return h

    def __y(self, h):

        return self.__output_fnc(self.__W_out, TT.dot(h[-1]) + self.__b_out)

