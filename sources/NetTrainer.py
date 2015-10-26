import time

import numpy

from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoElement import PrintableInfoElement
from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from Statistics import Statistics
from TrainingRule import TrainingRule

__author__ = 'giulio'


class NetTrainer(object):
    def __init__(self, training_rule: TrainingRule, obj_fnc: ObjectiveFunction):
        self.__training_rule = training_rule
        self.__obj_fnc = obj_fnc

        #  FIXME magic constants
        self.__max_it = 500000
        self.__batch_size = 100
        self.__validation_set_size = 10000
        self.__stop_error_thresh = 0.001
        self.__check_freq = 50

        self.__model_path = '/home/giulio/RNNs/models'
        self.__model_name = 'model'

        # build training setting info
        self.__trainign_settings_info = InfoGroup('settings',
                                                  InfoList(PrintableInfoElement('max_it', ':d', self.__max_it),
                                                           PrintableInfoElement('check_freq', ':d', self.__check_freq),
                                                           PrintableInfoElement('batch_size', ':d', self.__batch_size),
                                                           PrintableInfoElement('validation_set_size', ':d',
                                                                                self.__validation_set_size),
                                                           PrintableInfoElement('stop_error_thresh', ':f',
                                                                                self.__stop_error_thresh)
                                                           ))

    def train(self, task, activation_fnc, output_fnc, n_hidden, seed=13):

        # configure network
        net = RNN(activation_fnc, output_fnc, n_hidden, task.n_in, task.n_out, seed)

        # compile symbols
        train_step = self.__training_rule.compile(net, self.__obj_fnc)

        # training statistics
        stats = Statistics(self.__max_it, self.__check_freq)

        print('Generating validation set ...')
        validation_set = task.get_batch(self.__validation_set_size)
        print('... Done')

        rho = numpy.max(abs(numpy.linalg.eigvals(net.symbols.current_params.W_rec.get_value()))) #  FIXME
        print('Initial rho: {:5.2f}'.format(rho))

        print(self.__trainign_settings_info)
        print('Training ...\n')
        start_time = time.time()
        batch_start_time = time.time()

        i = 0
        best_error = 100
        while i < self.__max_it and best_error > self.__stop_error_thresh:

            batch = task.get_batch(self.__batch_size)
            train_info = train_step.step(batch.inputs, batch.outputs)

            if i % self.__check_freq == 0:
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
                net.save_model(self.__model_path, self.__model_name, stats, self.__trainign_settings_info)

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
