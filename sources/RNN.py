import theano as T
import theano.tensor as TT
import numpy
import time
from configs import Configs

__author__ = 'giulio'


class RNN:
    def __init__(self, task, activation_fnc, output_fnc, loss_fnc, n_hidden, seed):
        # topology
        self.__n_hidden = n_hidden
        self.__n_in = task.n_in
        self.__n_out = task.n_out

        # activation functions
        self.__activation_fnc = activation_fnc

        # output function
        self.__output_fnc = output_fnc

        # loss function
        self.__loss_fnc = loss_fnc

        # task
        self.__task = task

        # random generator
        self.__rng = numpy.random.RandomState(seed)

        # init weight matrices TODO
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
        self.__b_rec = T.shared(b_rec, 'b_rec', broadcastable=(False, True))
        self.__b_out = T.shared(b_out, 'b_out', broadcastable=(False, True))

        # define net output fnc
        u = TT.tensor3()
        n_sequences = u.shape[2]
        h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.__n_hidden, n_sequences)
        h = self.__h(h_m1, u)
        y = self.__y(h)
        self.__net_output = T.function([u], [y])

        # define gradient function
        t = TT.tensor3()
        loss = self.__loss_fnc(y, t)
        gW_rec, gW_in, gW_out, \
        gb_rec, gb_out = TT.grad(loss, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
        self.__gradient = T.function([u, t], [gW_rec, gW_in, gW_out, gb_rec, gb_out])

        # define train step
        gW_rec = TT.matrix()
        gW_in = TT.matrix()
        gW_out = TT.matrix()
        gb_rec = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True))
        gb_out = TT.tensor(dtype=Configs.floatType, broadcastable=(False, True))

        lr = TT.scalar()

        norm_theta = TT.sqrt((gW_rec ** 2).sum() +
                             (gW_in ** 2).sum() +
                             (gW_out ** 2).sum() +
                             (gb_rec ** 2).sum() +
                             (gb_out ** 2).sum())

        self.__train_step = T.function([gW_rec, gW_in, gW_out, gb_rec, gb_out, lr], [norm_theta],
                                       on_unused_input='warn',
                                       updates=[(self.__W_rec, self.__W_rec - lr * gW_rec),
                                                (self.__W_in, self.__W_in - lr * gW_in),
                                                (self.__W_out, self.__W_out - lr * gW_out),
                                                (self.__b_rec, self.__b_rec - lr * gb_rec),
                                                (self.__b_out, self.__b_out - lr * gb_out)])

    def net_output(self, sequence):
        return self.__net_output(sequence)[0]

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

    def train(self):

        # TODO move somewhere else
        lr = 0.01
        batch_size = 100

        start_time = time.time()
        for i in range(0, 1000):

            print('iteration num {}'.format(i))

            batch = self.__task.get_batch(batch_size)
            loss = self.__loss_fnc(self.net_output(batch.inputs), batch.outputs)


            gW_rec, gW_in, gW_out, \
            gb_rec, gb_out = self.__gradient(batch.inputs, batch.outputs)
            norm = self.__train_step(gW_rec, gW_in, gW_out, gb_rec, gb_out, lr)
            print('\tnorm = {}, loss = {}'.format(norm, loss))
            new_loss = self.__loss_fnc(self.net_output(batch.inputs), batch.outputs)
            print('\tloss after update = {}'.format(new_loss))

            rho =numpy.max(abs(numpy.linalg.eigvals(self.__W_rec.get_value())))
            print("\t rho = {}".format(rho))

        end_time = time.time()
        print('Elapsed time: {:2.2f}'.format(end_time-start_time))





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

    # predefined loss functions
    def squared_error(y, t):
        return ((t[-1:, :, :] - y[-1:, :, :]) ** 2).mean(axis=1).sum()
