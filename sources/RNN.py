import theano as T
import theano.tensor as TT
import numpy
import time
import os
from Configs import Configs
from SymbolsCloset import SymbolsCloset
from TrainingRule import TrainingRule

__author__ = 'giulio'


class RNN(object):
    def __init__(self, task, activation_fnc, output_fnc, loss_fnc, n_hidden, training_rule: TrainingRule, seed=13):

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

        # training rule
        self.__train_rule = training_rule

        # task
        self.__task = task

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
        self.symb_closet = SymbolsCloset(self, W_rec, W_in, W_out, b_rec, b_out, loss_fnc)

        # define visible functions
        self.net_output_shared = T.function([self.symb_closet.u], self.symb_closet.y_shared)

        # cosine = TT.dot(penalty_grad.flatten(), symbol_closet.gW_rec.flatten()) / (
        #   penalty_norm * TT.sqrt((symbol_closet.gW_rec ** 2).sum()))

        # define train step
        self.__train_step = self.__train_rule.get_train_step_fnc(self.symb_closet)

    @property
    def n_hidden(self):
        return self.__n_hidden

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    def net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
        h, deriv_a = self.__h(u, W_rec, W_in, b_rec)
        return self.__y(h, W_out, b_out), deriv_a

    def __h(self, u, W_rec, W_in, b_rec):
        def h_t(u_t, h_tm1, W_rec, W_in, b_rec):
            a_t = TT.dot(W_rec, h_tm1) + TT.dot(W_in, u_t) + b_rec
            deriv_a = self.__activation_fnc.grad_f(a_t)
            return self.__activation_fnc.f(a_t), deriv_a

        n_sequences = u.shape[2]
        h_m1 = TT.alloc(numpy.array(0., dtype=Configs.floatType), self.__n_hidden, n_sequences)

        values, _ = T.scan(h_t, sequences=u,
                           outputs_info=[h_m1, None],
                           non_sequences=[W_rec, W_in, b_rec],
                           name='h_t',
                           mode=T.Mode(linker='cvm'))
        a = values[1]
        h = values[0]
        return h, a

    # single output mode
    # def __y(self, h):
    #     return self.__output_fnc(TT.dot(self.__W_out, h[-1]) + self.__b_out)

    def __y(self, h, W_out, b_out):

        def y_t(h_t, W_out, b_out):
            return self.__output_fnc(TT.dot(W_out, h_t) + b_out)

        y, _ = T.scan(y_t, sequences=h,
                      outputs_info=[None],
                      non_sequences=[W_out, b_out],
                      name='y_t',
                      mode=T.Mode(linker='cvm'))
        return y

    def save_model(self, path, filename, stats):
        """saves the model with statistics to file"""

        os.makedirs(path, exist_ok=True)

        numpy.savez(path + '/' + filename + '.npz',
                    n_hidden=self.__n_hidden,
                    n_in=self.__n_in,
                    n_out=self.__n_out,
                    task=str(self.__task),
                    activation_fnc=str(self.__activation_fnc),
                    valid_error=stats.valid_error,
                    gradient_norm=stats.grad_norm,
                    rho=stats.rho,
                    penalty=stats.penalty_norm,
                    elapsed_time=stats.elapsed_time,
                    W_rec=self.symb_closet.W_rec.get_value(),
                    W_in=self.symb_closet.W_in.get_value(),
                    W_out=self.symb_closet.W_out.get_value(),
                    b_rec=self.symb_closet.b_rec.get_value(),
                    b_out=self.symb_closet.b_out.get_value())

    def train(self):

        # TODO move somewhere else and add them to npz saved file
        max_it = 500000
        batch_size = 100
        validation_set_size = 10000
        stop_error_thresh = 0.001
        check_freq = 50

        model_path = '/home/giulio/RNNs/models'
        model_name = 'model'

        # training statistics
        stats = Statistics(max_it, check_freq)

        print('Generating validation set...')
        validation_set = self.__task.get_batch(validation_set_size)

        rho = numpy.max(abs(numpy.linalg.eigvals(self.symb_closet.W_rec.get_value())))
        print('Initial rho: {:5.2f}'.format(rho))

        print('Training...')
        start_time = time.time()
        batch_start_time = time.time()

        i = 0
        best_error = 100
        while i < max_it and best_error > stop_error_thresh:

            batch = self.__task.get_batch(batch_size)
            infos = self.__train_step(batch.inputs, batch.outputs)

            norm = infos[0]
            penalty_grad_norm = infos[1]

            if i % check_freq == 0:
                y_net = self.net_output_shared(validation_set.inputs)
                valid_error = self.__task.error_fnc(y_net, validation_set.outputs)
                loss = self.__loss_fnc(y_net, validation_set.outputs)
                rho = numpy.max(abs(numpy.linalg.eigvals(self.symb_closet.W_rec.get_value())))

                self.save_model(model_path, model_name, stats)

                if valid_error < best_error:
                    best_error = valid_error

                batch_end_time = time.time()
                total_elapsed_time = batch_end_time - start_time
                stats.update(rho, norm, penalty_grad_norm, valid_error, i, total_elapsed_time)

                print(
                    'iteration: {:07d}, grad norm: {:07.3f}, valid loss: {:07.3f}, valid error: {:.2%} '
                    '(best: {:.2%}), rho: {:5.2f}, '.format(
                        i, norm.item(), loss, valid_error, best_error, rho) + self.__train_rule.format_infos(
                        infos) + ', time: {:2.2f}'.format(
                        batch_end_time - batch_start_time))
                batch_start_time = time.time()

            i += 1

        end_time = time.time()
        print('Elapsed time: {:2.2f} min'.format((end_time - start_time) / 60))

    # predefined output functions
    @staticmethod
    def last_linear_fnc(y):
        return y

    # predefined loss functions
    @staticmethod
    def squared_error(y, t):
        return ((t[-1:, :, :] - y[-1:, :, :]) ** 2).sum(axis=0).mean()


class Statistics(object):
    def __init__(self, max_it, check_freq):
        self.__check_freq = check_freq
        self.__current_it = 0
        self.__actual_length = 0

        m = numpy.ceil(max_it / check_freq) - 1
        self.__rho_values = numpy.zeros((m,), dtype='float32')
        self.__grad_norm_values = numpy.zeros((m,), dtype='float32')
        self.__penalty_norm_values = numpy.zeros((m,), dtype='float32')
        self.__valid_error_values = numpy.zeros((m,), dtype='float32')
        self.__elapsed_time = 0

    def update(self, rho, grad_norm, penalty_grad_norm, valid_error, it, elapsed_time):
        j = it / self.__check_freq
        self.__current_it = it
        self.__actual_length += 1
        self.__rho_values[j] = rho
        self.__grad_norm_values[j] = grad_norm
        self.__penalty_norm_values[j] = penalty_grad_norm
        self.__valid_error_values[j] = valid_error
        self.__elapsed_time = elapsed_time

    @property
    def rho(self):
        return self.__rho_values[0:self.__actual_length]

    @property
    def grad_norm(self):
        return self.__grad_norm_values[0:self.__actual_length]

    @property
    def valid_error(self):
        return self.__valid_error_values[0:self.__actual_length]

    @property
    def penalty_norm(self):
        return self.__penalty_norm_values[0:self.__actual_length]

    @property
    def elapsed_time(self):
        return self.__elapsed_time
