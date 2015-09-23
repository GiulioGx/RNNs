import theano as T
import theano.tensor as TT
import numpy
from configs import Configs

__author__ = 'giulio'


class RNN:
    def __init__(self, task, activation_fnc, output_fnc, n_hidden, seed):

        # topology
        self.__n_hidden = n_hidden
        self.__n_in = task.n_in
        self.__n_out = task.n_out

        # activation fnc
        self.__activation_fnc = activation_fnc

        # output_fnc
        self.__output_fnc = output_fnc

        # task
        self.__task = task

        # random generator
        self.__rng = numpy.random.RandomState(seed)

        # init weight matrices
        W_in = numpy.asarray(
            self.__rng.normal(size=(self.__n_hidden, self.__n_in), scale=.01, loc=.0), dtype=Configs.floatType)
        W_rec = numpy.asarray(
            self.__rng.normal(size=(self.__n_hidden, self.__n_hidden), scale=.01, loc=.0), dtype=Configs.floatType)
        W_out = numpy.asarray(
            self.__rng.normal(size=(self.__n_out, self.__n_hidden), scale=.01, loc=0.0), dtype=Configs.floatType)

        # init biases
        b_rec = numpy.zeros((self.__n_hidden, 1), Configs.floatType)
        b_out = numpy.zeros((self.__n_out, 1), Configs.floatType)

        self.__W_in = T.shared(W_in, 'W_in')
        self.__W_rec = T.shared(W_rec, 'W_rec')
        self.__W_out = T.shared(W_out, 'W_out')
        self.__b_rec = T.shared(b_rec, 'b_rec')
        self.__b_out = T.shared(b_out, 'b_out')

        print(W_in.shape)

        # define net output fnc
        u = TT.tensor3()
        n_sequences = u.shape[2]
        h_m1 = TT.alloc(numpy.array(0, dtype=Configs.floatType), self.__n_hidden, n_sequences)
        h = self.__h(h_m1, u)
        y = self.__y(h)
        self.__net_output = T.function([u], [y])

    def net_output(self, sequence):
        return self.__net_output(sequence)

    def __h(self, h_m1, u):

        def h_t(u_t, h_tm1):
            return self.__activation_fnc(TT.dot(self.__W_rec, h_tm1) + TT.dot(self.__W_in, u_t) + self.__b_rec)

        h, _ = T.scan(h_t, sequences=u,
                      outputs_info=[h_m1],
                      non_sequences=[],
                      name='h_t',
                      mode=T.Mode(linker='cvm'))
        return h

    # single output mode
    # def __y(self, h):
    #     return self.__output_fnc(TT.dot(self.__W_out, h[-1]) + self.__b_out)

    def __y(self, h):

        def y_t(h_t):
            return self.__output_fnc(TT.dot(self.__W_out, h_t) + self.__b_out)

        y, _ = T.scan(y_t, sequences=h,
                      outputs_info=[None],
                      non_sequences=[],
                      name='y_t',
                      mode=T.Mode(linker='cvm'))
        return y

    # predefined activation functions
    def relu(x):
        return TT.switch(x < 0, 0, x)

    def sigmoid(x):
        return TT.nnet.sigmoid(x)

    def tanh(x):
        return TT.tanh(x)

    # predefined output functions
    def last_linear_fnc(y):
        return y





