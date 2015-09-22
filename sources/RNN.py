import theano as T
import theano.tensor as TT

__author__ = 'giulio'


class RNN:
    def __init__(self, task, activation_fnc, n_hidden):

        self.__n_hidden = n_hidden
        self.__task = task
        self.__activation_fnc = activation_fnc

        W_uh = numpy.asarray(
            self.__rng.normal(size=(self.__nin, self.__n_hidden), scale=.01, loc=.0), dtype=self.__floatX)
        W_hh = numpy.asarray(
            self.__rng.normal(size=(self.__n_hidden, self.__n_hidden), scale=.01, loc=.0), dtype=self.__floatX)
        W_hy = numpy.asarray(
            self.__rng.normal(size=(self.__n_hidden, self.__nout), scale=.01, loc=0.0), dtype=self.__floatX)
        b_hh = numpy.zeros((self.__n_hidden,), dtype=self.__floatX)
        b_hy = numpy.zeros((self.__nout,), dtype=self.__floatX)

        self.__W_uh = T.shared(W_uh, 'W_uh')
        self.__W_hh = T.shared(W_hh, 'W_hh')
        self.__W_hy = T.shared(W_hy, 'W_hy')
        self.__b_hh = T.shared(b_hh, 'b_hh')
        self.__b_hy = T.shared(b_hy, 'b_hy')

    def __get_h(self, h0_tm1, u):

        def h_t(u_t, h_tm1, W_hh, W_uh, W_hy):
            return self.__activation_fnc(TT.dot(h_tm1, W_hh) + TT.dot(u_t, W_uh) + self.__b_hh)

        h, _ = T.scan(h_t, sequences=u,
                      outputs_info=[h0_tm1],
                      non_sequences=[self.__W_hh, self.__W_uh, self.__W_hy],
                      name='recurrent_fn',
                      mode=theano.Mode(linker='cvm'))
        return h
