import time

import numpy
from Infos import PrintableInfoElement, InfoGroup, InfoList

from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from Statistics import Statistics
from TrainingRule import TrainingRule

__author__ = 'giulio'


class NetTrainer(object):
    def __init__(self, training_rule: TrainingRule, obj_fnc: ObjectiveFunction):
        self.__training_rule = training_rule
        self.__obj_fnc = obj_fnc

    def train(self, task, activation_fnc, output_fnc, n_hidden, seed=13):

        # configure network
        net = RNN(activation_fnc, output_fnc, n_hidden, task.n_in, task.n_out, seed)

        # compile symbols
        train_step = self.__training_rule.compile(net, self.__obj_fnc)

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

        rho = numpy.max(abs(numpy.linalg.eigvals(net.symbols.current_params.W_rec.get_value())))
        print('Initial rho: {:5.2f}'.format(rho))

        print('Training...')
        start_time = time.time()
        batch_start_time = time.time()

        i = 0
        best_error = 100
        while i < max_it and best_error > stop_error_thresh:

            batch = task.get_batch(batch_size)
            train_info = train_step.step(batch.inputs, batch.outputs)

            if i % check_freq == 0:
                y_net = net.net_output_shared(validation_set.inputs)  # FIXME
                valid_error = task.error_fnc(y_net, validation_set.outputs)
                valid_loss = self.__obj_fnc.loss(y_net, validation_set.outputs)
                rho = numpy.max(abs(numpy.linalg.eigvals(net.symbols.current_params.W_rec.get_value())))

                if valid_error < best_error:
                    best_error = valid_error

                batch_end_time = time.time()
                total_elapsed_time = batch_end_time - start_time

                batch_time = batch_end_time - batch_start_time
                info = self.__build_infos(train_info, i, valid_loss, valid_error, best_error, rho, batch_time)
                print(info)
                stats.update(info, i, total_elapsed_time)
                net.save_model(model_path, model_name, stats)

                batch_start_time = time.time()

            i += 1

        end_time = time.time()
        print('Elapsed time: {:2.2f} min'.format((end_time - start_time) / 60))
        return net

    def __build_infos(self, train_info, i, valid_loss, valid_error, best_error, rho, batch_time):

        it_info = PrintableInfoElement('iteration', ':07d', i)

        val_loss_info = PrintableInfoElement('loss', ':07.3f', valid_loss)

        error_info = PrintableInfoElement('curr', ':.2%', valid_error)
        best_info = PrintableInfoElement('best', ':.2%', best_error)
        error_group = InfoGroup('error', InfoList(error_info, best_info))

        val_info = InfoGroup('validation', InfoList(val_loss_info, error_group))

        rho_info = PrintableInfoElement('rho', ':5.2f', rho)
        time_info = PrintableInfoElement('time', ':2.2f', batch_time)

        info = InfoList(it_info, val_info, rho_info, train_info, time_info)
        return info

    # predefined loss functions
    @staticmethod
    def squared_error(y, t):
        return ((t[-1:, :, :] - y[-1:, :, :]) ** 2).sum(axis=0).mean()
