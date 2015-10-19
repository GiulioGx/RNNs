import time

import numpy

from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from TrainingRule import TrainingRule

__author__ = 'giulio'


class RNNTrainer(object):

    def __init__(self, training_rule: TrainingRule, obj_fnc: ObjectiveFunction):
        self.__training_rule = training_rule
        self.__obj_fnc = obj_fnc

    def train(self, task, activation_fnc, output_fnc, n_hidden, seed=13):

        net = RNN(activation_fnc, output_fnc, n_hidden, task.n_in, task.n_out, seed)
        obj_symbols = self.__obj_fnc.obj_symbols(net)

        #define train step
        train_step = self.__training_rule.get_train_step_fnc(net.symb_closet, obj_symbols)

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
        validation_set = task.get_batch(validation_set_size)

        rho = numpy.max(abs(numpy.linalg.eigvals(net.symb_closet.W_rec.get_value())))
        print('Initial rho: {:5.2f}'.format(rho))

        print('Training...')
        start_time = time.time()
        batch_start_time = time.time()

        i = 0
        best_error = 100
        while i < max_it and best_error > stop_error_thresh:

            batch = task.get_batch(batch_size)
            infos = train_step(batch.inputs, batch.outputs)

            norm = infos[0]
            penalty_grad_norm = infos[1]

            if i % check_freq == 0:
                y_net = net.net_output_shared(validation_set.inputs) # FIXME
                valid_error = task.error_fnc(y_net, validation_set.outputs)
                loss = self.__obj_fnc.loss(y_net, validation_set.outputs)
                rho = numpy.max(abs(numpy.linalg.eigvals(net.symb_closet.W_rec.get_value())))

                net.save_model(model_path, model_name, stats)

                if valid_error < best_error:
                    best_error = valid_error

                batch_end_time = time.time()
                total_elapsed_time = batch_end_time - start_time
                stats.update(rho, norm, penalty_grad_norm, valid_error, i, total_elapsed_time)

                print(
                    'iteration: {:07d}, valid loss: {:07.3f}, valid error: {:.2%} '
                    '(best: {:.2%}), rho: {:5.2f}, '.format(
                        i, loss, valid_error, best_error, rho) + self.__training_rule.format_infos(
                        infos) + ', time: {:2.2f}'.format(
                        batch_end_time - batch_start_time))
                batch_start_time = time.time()

            i += 1

        end_time = time.time()
        print('Elapsed time: {:2.2f} min'.format((end_time - start_time) / 60))
        return net

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
