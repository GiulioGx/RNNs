import time
import numpy
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoElement import PrintableInfoElement
from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from Statistics import Statistics
from TrainingRule import TrainingRule
from initialization.GaussianInit import GaussianInit
from initialization.MatrixInit import MatrixInit
import logging
import os

__author__ = 'giulio'


class NetTrainer(object):
    def __init__(self, training_rule: TrainingRule, obj_fnc: ObjectiveFunction,
                 init_strategy: MatrixInit = GaussianInit(), max_it=500000, bacth_size=100, validation_set_size=10000,
                 stop_error_thresh=0.001, check_freq=50, model_save_file='./model', log_filename='./model.log'):

        self.__training_rule = training_rule
        self.__obj_fnc = obj_fnc
        self.__init_strategy = init_strategy
        self.__max_it = max_it
        self.__batch_size = bacth_size
        self.__validation_set_size = validation_set_size
        self.__stop_error_thresh = stop_error_thresh
        self.__check_freq = check_freq
        self.__model_filename = model_save_file
        self.__log_filename = log_filename

        # logging
        if os.path.exists(log_filename):
            os.remove(log_filename)
        logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(levelname)s:%(message)s')

        # build training setting info
        self.__training_settings_info = InfoGroup('settings',
                                                  InfoList(PrintableInfoElement('max_it', ':d', self.__max_it),
                                                           PrintableInfoElement('check_freq', ':d', self.__check_freq),
                                                           PrintableInfoElement('batch_size', ':d', self.__batch_size),
                                                           PrintableInfoElement('validation_set_size', ':d',
                                                                                self.__validation_set_size),
                                                           PrintableInfoElement('stop_error_thresh', ':f',
                                                                                self.__stop_error_thresh),
                                                           ))

    def train(self, task, activation_fnc, output_fnc, n_hidden, seed=13):

        # add task description to infos
        self.__training_settings_info = InfoList(self.__training_settings_info,
                                                 PrintableInfoElement('task', '', str(task)))

        # configure network
        net = RNN(activation_fnc, output_fnc, n_hidden, task.n_in, task.n_out, self.__init_strategy, seed)

        # compile symbols
        train_step = self.__training_rule.compile(net, self.__obj_fnc)

        # training statistics
        stats = Statistics(self.__max_it, self.__check_freq)

        logging.info('Generating validation set ...')
        validation_set = task.get_batch(self.__validation_set_size)
        logging.info('... Done')

        rho = numpy.max(abs(numpy.linalg.eigvals(net.symbols.current_params.W_rec.get_value())))  # FIXME
        logging.info('Initial rho: {:5.2f}'.format(rho))

        logging.info(self.__training_settings_info)
        logging.info('Training ...\n')
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
                info = NetTrainer.__build_infos(train_info, i, valid_loss, valid_error, best_error, rho, batch_time)
                logging.info(info)
                stats.update(info, i, total_elapsed_time)
                net.save_model(self.__model_filename, stats, self.__training_settings_info)

                batch_start_time = time.time()
            i += 1

        end_time = time.time()
        logging.info('Elapsed time: {:2.2f} min'.format((end_time - start_time) / 60))
        return net

    @staticmethod
    def __build_infos(train_info, i, valid_loss, valid_error, best_error, rho, batch_time):

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
