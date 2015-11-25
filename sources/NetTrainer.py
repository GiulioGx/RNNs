import logging
import os
import time

import numpy
import theano as T
from numpy.linalg import LinAlgError

from ObjectiveFunction import ObjectiveFunction
from Statistics import Statistics
from TrainingRule import TrainingRule
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from model.RNN import RNN

__author__ = 'giulio'


class NetTrainer(object):
    def __init__(self, training_rule: TrainingRule, obj_fnc: ObjectiveFunction, max_it=10 ** 5, bacth_size=100,
                 validation_set_size=10 ** 4,
                 stop_error_thresh=0.01, check_freq=50, output_dir='.'):

        self.__training_rule = training_rule
        self.__obj_fnc = obj_fnc
        self.__max_it = max_it
        self.__batch_size = bacth_size
        self.__validation_set_size = validation_set_size
        self.__stop_error_thresh = stop_error_thresh
        self.__check_freq = check_freq
        self.__output_dir = output_dir

        # logging
        log_filename = self.__output_dir + '/train.log'
        if os.path.exists(log_filename):
            os.remove(log_filename)
        os.makedirs(output_dir, exist_ok=True)
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

    def _train(self, task, net):
        # add task description to infos
        self.__training_settings_info = InfoList(self.__training_settings_info,
                                                 PrintableInfoElement('task', '', str(task)))

        # compile symbols
        logging.info('Compiling theano functions for the training step...')
        train_step = self.__training_rule.compile(net, self.__obj_fnc)
        logging.info('... Done')

        #  loss and error theano fnc
        u = net.symbols.u
        t = net.symbols.t
        y = net.symbols.y_shared
        error = task.error_fnc(y, t)
        loss = self.__obj_fnc.loss(y, t)
        self.__loss_and_error = T.function([u, t], [error, loss], name='loss_and_error_fnc')

        # training statistics
        stats = Statistics(self.__max_it, self.__check_freq)

        logging.info('Generating validation set ...')
        validation_set = task.get_batch(self.__validation_set_size)
        logging.info('... Done')

        logging.info(str(net.info))

        logging.info(str(self.__training_settings_info))
        logging.info(str(self.__training_rule.infos))
        logging.info('Training ...\n')
        start_time = time.time()
        batch_start_time = time.time()

        error_occured = False
        i = 0
        best_error = 100
        while i < self.__max_it and best_error > self.__stop_error_thresh / 100 and (not error_occured):

            batch = task.get_batch(self.__batch_size)
            train_info = train_step.step(batch.inputs, batch.outputs)

            if i % self.__check_freq == 0:
                eval_start_time = time.time()
                valid_error, valid_loss = self.__loss_and_error(validation_set.inputs, validation_set.outputs)
                valid_error = valid_error.item()
                valid_loss = valid_loss.item()

                try:
                    rho = net.spectral_radius
                except LinAlgError as e:
                    logging.error(str(e))
                    error_occured = True
                    rho = numpy.nan

                if valid_error < best_error:
                    best_error = valid_error

                batch_end_time = time.time()
                total_elapsed_time = batch_end_time - start_time

                eval_time = time.time() - eval_start_time
                batch_time = batch_end_time - batch_start_time

                info = NetTrainer.__build_infos(train_info, i, valid_loss, valid_error, best_error, rho,
                                                batch_time, eval_time)
                logging.info(info)
                stats.update(info, i, total_elapsed_time)
                model_filename = self.__output_dir + '/model'
                net.save_model(model_filename, stats, self.__training_settings_info)

                batch_start_time = time.time()
                if error_occured:
                    logging.warning('stopping training...')

            i += 1

        end_time = time.time()
        if i == self.__max_it:
            logging.warning('Maximum number of iterations reached, stopping training...')
        elif best_error <= self.__stop_error_thresh / 100:
            logging.info('Training succeded, validation error below the given threshold({:.2%})'.format(
                self.__stop_error_thresh / 100))
        logging.info('Elapsed time: {:2.2f} min'.format((end_time - start_time) / 60))
        return net

    def train(self, task, activation_fnc, output_fnc, n_hidden, init_strategies=RNN.deafult_init_strategies, seed=13):

        # configure network
        logging.info('Compiling theano functions for the net...')
        net = RNN(activation_fnc, output_fnc, n_hidden, task.n_in, task.n_out, init_strategies, seed)
        logging.info('...Done')
        return self._train(task, net)

    def resume_training(self, task, net):  # TODO load statistics too
        return self._train(task, net)

    @staticmethod
    def __build_infos(train_info, i, valid_loss, valid_error, best_error, rho, batch_time, eval_time):

        it_info = PrintableInfoElement('iteration', ':07d', i)

        val_loss_info = PrintableInfoElement('loss', ':07.4f', valid_loss)

        error_info = PrintableInfoElement('curr', ':.2%', valid_error)
        best_info = PrintableInfoElement('best', ':.2%', best_error)
        error_group = InfoGroup('error', InfoList(error_info, best_info))

        val_info = InfoGroup('validation', InfoList(val_loss_info, error_group))

        rho_info = PrintableInfoElement('rho', ':5.2f', rho)
        eval_time_info = PrintableInfoElement('eval', ':2.2f', eval_time)
        batch_time_info = PrintableInfoElement('tot', ':2.2f', batch_time)
        time_info = InfoGroup('time', InfoList(eval_time_info, batch_time_info))

        info = InfoList(it_info, val_info, rho_info, train_info, time_info)
        return info

    # predefined loss functions
    @staticmethod
    def squared_error(y, t):
        return ((t[-1:, :, :] - y[-1:, :, :]) ** 2).sum(axis=0).mean()

        # @staticmethod
        # def cross_entropy(y, t):
        #     return -(t[-1, :, :] * TT.log(y[-1, :, :])).sum(axis=0).mean()
