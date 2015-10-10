import theano as T
import theano.tensor as TT
import numpy
import time
import os
from configs import Configs
from Penalty import NullPenalty

__author__ = 'giulio'


class RNN(object):
    def __init__(self, task, activation_fnc, output_fnc, loss_fnc, n_hidden, penalty=NullPenalty(), seed=13):

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

        # penalty function
        self.__penalty_expr = penalty

        # task
        self.__task = task

        # random generator
        self.__rng = numpy.random.RandomState(seed)

        # init weight matrices TODO
        scale = .1
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

        u = TT.tensor3()  # labels

        y, deriv_a = self.__net_output(W_rec, W_in, W_out, b_rec, b_out, u)
        y_shared, deriv_a_shared = T.clone([y, deriv_a],
                                           replace=[(W_rec, self.__W_rec), (W_in, self.__W_in), (W_out, self.__W_out),
                                                    (b_rec, self.__b_rec), (b_out, self.__b_out)])
        self.net_output = T.function([u], y_shared)

        # define (shared) gradient function
        t = TT.tensor3()
        loss_shared = self.__loss_fnc(y_shared, t)
        gW_rec, gW_in, gW_out, \
        gb_rec, gb_out = TT.grad(loss_shared, [self.__W_rec, self.__W_in, self.__W_out, self.__b_rec, self.__b_out])
        grad_norm = TT.sqrt((gW_rec ** 2).sum() +
                            (gW_in ** 2).sum() +
                            (gW_out ** 2).sum() +
                            (gb_rec ** 2).sum() +
                            (gb_out ** 2).sum())
        # self.__gradient = T.function([u, t], [gW_rec, gW_in, gW_out, gb_rec, gb_out, grad_norm])

        # descent direction (anti-gradient for now)
        W_rec_dir = -gW_rec
        W_in_dir = -gW_in
        W_out_dir = -gW_out
        b_rec_dir = -gb_rec
        b_out_dir = -gb_out

        # add penalty term
        penalty_value, penalty_grad = self.__penalty_expr.penalty_term(deriv_a_shared, self.__W_rec)
        penalty_norm = TT.sqrt((penalty_grad ** 2).sum())
        penalty_grad = TT.cast(penalty_grad, dtype=Configs.floatType)
        penalty_norm = TT.cast(penalty_norm, dtype=Configs.floatType)
        W_rec_dir = TT.switch(penalty_norm > 0, W_rec_dir - (
            TT.alloc(numpy.array(0.003, dtype=Configs.floatType)) * penalty_grad / penalty_norm), W_rec_dir)

        self._penalty_debug = T.function([u], [deriv_a_shared, penalty_value, penalty_norm])  # FIXME debug


        # TODO move somewhere else
        # normalized constant step
        lr = TT.alloc(numpy.array(0.01, dtype=Configs.floatType))
        lr = lr / grad_norm
        n_steps = TT.alloc(numpy.array(0, dtype=int))

        max_steps = 10

        gradient = TT.concatenate(
            [gW_rec.flatten(), gW_in.flatten(), gW_out.flatten(), gb_rec.flatten(), gb_out.flatten()]).flatten()

        direction = TT.concatenate(
            [W_rec_dir.flatten(), W_in_dir.flatten(), W_out_dir.flatten(), b_rec_dir.flatten(),
             b_out_dir.flatten()]).flatten()
        grad_dir_dot_product = TT.dot(gradient, direction)

        def armijo_step(step, beta, alpha, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir, gW_rec, gW_in, gW_out,
                        gb_rec,
                        gb_out, f_0, u, t, grad_dir_dot_product):
            W_rec_k = self.__W_rec + step * W_rec_dir
            W_in_k = self.__W_in + step * W_in_dir
            W_out_k = self.__W_out + step * W_out_dir
            b_rec_k = self.__b_rec + step * b_rec_dir
            b_out_k = self.__b_out + step * b_out_dir

            f_1 = self.__loss(W_rec_k, W_in_k, W_out_k, b_rec_k, b_out_k, u, t)

            condition = f_0 - f_1 >= -alpha * step * grad_dir_dot_product  # sufficient decrease condition

            return step * beta, [], T.scan_module.until(
                condition)

        # TODO move somewhere else (armijo parameters)
        step = TT.alloc(numpy.array(1, dtype=Configs.floatType))
        beta = TT.alloc(numpy.array(0.5, dtype=Configs.floatType))
        alpha = TT.alloc(numpy.array(0.1, dtype=Configs.floatType))

        values, updates = T.scan(armijo_step, outputs_info=step,
                                 non_sequences=[beta, alpha, W_rec_dir, W_in_dir, W_out_dir, b_rec_dir, b_out_dir,
                                                gW_rec, gW_in, gW_out,
                                                gb_rec,
                                                gb_out, loss_shared, u, t, grad_dir_dot_product], n_steps=max_steps)

        # lr = values[-1] / beta
        # n_steps = values.size

        # define train step
        self.__train_step = T.function([u, t], [grad_norm, lr, n_steps, penalty_norm],
                                       allow_input_downcast='true',
                                       on_unused_input='warn',
                                       updates=[(self.__W_rec, self.__W_rec + lr * W_rec_dir),
                                                (self.__W_in, self.__W_in + lr * W_in_dir),
                                                (self.__W_out, self.__W_out + lr * W_out_dir),
                                                (self.__b_rec, self.__b_rec + lr * b_rec_dir),
                                                (self.__b_out, self.__b_out + lr * b_out_dir)])

    # FIXME debug
    @property
    def W_rec(self):
        return self.__W_rec.get_value()

    @property
    def n_hidden(self):
        return self.__n_hidden

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    def __net_output(self, W_rec, W_in, W_out, b_rec, b_out, u):
        h, deriv_a = self.__h(u, W_rec, W_in, b_rec)
        return self.__y(h, W_out, b_out), deriv_a

    def __loss(self, W_rec, W_in, W_out, b_rec, b_out, u, t):
        y, _ = self.__net_output(W_rec, W_in, W_out, b_rec, b_out, u)
        return self.__loss_fnc(y, t)

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
                    W_rec=self.__W_rec.get_value(),
                    W_in=self.__W_in.get_value(),
                    W_out=self.__W_out.get_value(),
                    b_rec=self.__b_rec.get_value(),
                    b_out=self.__b_out.get_value())

    def train(self):
        # TODO move somewhere else and add them to npz saved file
        max_it = 500000
        batch_size = 100
        validation_set_size = 10000
        #stop_error_thresh = 0.001
        stop_error_thresh = 0.36
        check_freq = 50

        model_path = '/home/giulio/RNNs/models'
        model_name = 'model'

        # training statistics
        stats = Statistics(max_it, check_freq)

        print('Generating validation set...')
        validation_set = self.__task.get_batch(validation_set_size)

        rho = numpy.max(abs(numpy.linalg.eigvals(self.__W_rec.get_value())))
        print('Initial rho: {:5.2f}'.format(rho))

        print('Training...')
        start_time = time.time()
        batch_start_time = time.time()

        i = 0
        best_error = 100
        while i < max_it and best_error > stop_error_thresh:

            batch = self.__task.get_batch(batch_size)
            norm, lr, n_steps, penalty_grad_norm = self.__train_step(batch.inputs, batch.outputs)

            if i % check_freq == 0:
                y_net = self.net_output(validation_set.inputs)
                valid_error = self.__task.error_fnc(y_net, validation_set.outputs)
                loss = self.__loss_fnc(y_net, validation_set.outputs)
                rho = numpy.max(abs(numpy.linalg.eigvals(self.__W_rec.get_value())))

                self.save_model(model_path, model_name, stats)

                if valid_error < best_error:
                    best_error = valid_error

                batch_end_time = time.time()
                total_elapsed_time = batch_end_time-start_time
                stats.update(rho, norm, penalty_grad_norm, valid_error, i, total_elapsed_time )

                print('iteration {:07d}: grad norm = {:07.3f}, valid loss = {:07.3f},'
                      ' valid error = {:.2%} (best: {:.2%}), rho = {:5.2f}, penalty = {:07.3f},'
                      ' lr = {:02.4f}, n_steps = {:02d} time  ={:2.2f}'
                      .format(i, norm.item(), loss, valid_error, best_error, rho, penalty_grad_norm.item(), lr.item(),
                              n_steps.item(),
                              batch_end_time - batch_start_time))
                batch_start_time = time.time()

            i += 1

        end_time = time.time()
        print('Elapsed time: {:2.2f} min'.format((end_time - start_time) / 60))

    # predefined output functions
    def last_linear_fnc(y):
        return y

    # predefined loss functions
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
