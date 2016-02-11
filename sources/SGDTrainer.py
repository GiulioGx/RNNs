import datetime
import logging
import os
import time

import numpy
import theano as T
from numpy.linalg import LinAlgError

from ActivationFunction import ActivationFunction
from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from Statistics import Statistics
from TrainingRule import TrainingRule
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from lossFunctions import LossFunction
from model import NetManager
from model.RNN import RNN
from model.RNNManager import RNNManager
from output_fncs.OutputFunction import OutputFunction
from task import Batch
from task.BatchPolicer import RepetitaPolicer
from task.Dataset import Dataset

__author__ = 'giulio'


class SGDTrainer(object):
    def __init__(self, training_rule: TrainingRule, max_it=10 ** 5, batch_size=100,
                 stop_error_thresh=0.01, check_freq=50, output_dir='.', incremental_units: bool = False):

        self.__training_rule = training_rule
        self.__max_it = max_it
        self.__batch_size = batch_size
        self.__stop_error_thresh = stop_error_thresh
        self.__check_freq = check_freq
        self.__output_dir = output_dir
        self.__incremental_units = incremental_units

        self.__log_filename = self.__output_dir + '/train.log'

        # build training setting info
        self.__training_settings_info = InfoGroup('settings',
                                                  InfoList(PrintableInfoElement('max_it', ':d', self.__max_it),
                                                           PrintableInfoElement('check_freq', ':d', self.__check_freq),
                                                           PrintableInfoElement('batch_size', ':d', self.__batch_size),
                                                           PrintableInfoElement('stop_error_thresh', ':f',
                                                                                self.__stop_error_thresh)
                                                           ))

    def _train(self, dataset: Dataset, net_manager: NetManager):

        net = net_manager.get_net(n_in=dataset.n_in, n_out=dataset.n_out)

        # add task description to infos
        self.__training_settings_info = InfoList(self.__training_settings_info,
                                                 PrintableInfoElement('task', '', str(dataset)))

        # compute_update symbols
        logging.info('Compiling theano functions for the training step...')
        train_step = self.__training_rule.compile(net)
        logging.info('... Done')
        logging.info('Seed: {}'.format(Configs.seed))

        #  loss and error theano fnc
        u = net.symbols.u
        t = net.symbols.t
        y = net.symbols.y_shared
        error = dataset.computer_error(y, t)
        loss = self.__training_rule.loss_fnc.value(y, t)
        self.__loss_and_error = T.function([u, t], [error, loss], name='loss_and_error_fnc')

        # training statistics
        stats = Statistics(self.__max_it, self.__check_freq, self.__training_settings_info, net.info)

        # policer #TODO se funziona mettere a pulito
        policer = RepetitaPolicer(dataset=dataset, batch_size=self.__batch_size, n_repetitions=100, block_size=1000)

        logging.info('Generating validation set ...')
        validation_set = dataset.validation_set
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

        while i < self.__max_it and best_error > self.__stop_error_thresh / 100 and (
                not error_occured):  # FOXME strategy criterio d'arresto

            # this makes the network grow in size according to the policy
            # specified when instanziating the network manager (may be even a null grow)
            net_manager.grow_net()

            batch = dataset.get_train_batch(self.__batch_size)
            # batch = policer.get_train_batch()
            train_info = train_step.step(batch.inputs, batch.outputs)

            if i % self.__check_freq == 0:  # FIXME 1st iteration
                eval_start_time = time.time()
                valid_error, valid_loss = SGDTrainer.compute_error_and_loss(self.__loss_and_error, validation_set)

                try:
                    rho = net.spectral_radius
                except LinAlgError as e:
                    logging.error(str(e))
                    error_occured = True
                    rho = numpy.nan

                if not error_occured:

                    if valid_error < best_error:
                        best_error = valid_error
                        net.save_model(self.__output_dir + '/best_model')

                    batch_end_time = time.time()
                    total_elapsed_time = batch_end_time - start_time

                    eval_time = time.time() - eval_start_time
                    batch_time = batch_end_time - batch_start_time

                    info = SGDTrainer.__build_infos(train_info, i, valid_loss, valid_error, best_error, rho,
                                                    batch_time, eval_time)
                    logging.info(info)
                    stats.update(info, i, total_elapsed_time)
                    net.save_model(self.__output_dir + '/current_model')
                    stats.save(self.__output_dir + '/stats')

                    batch_start_time = time.time()
                else:

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

    def __start_logger(self):
        os.makedirs(self.__output_dir, exist_ok=True)
        logging.basicConfig(filename=self.__log_filename, level=logging.INFO, format='%(levelname)s:%(message)s')
        now = datetime.datetime.now()
        logging.info('starting logging activity in date {}'.format(now.strftime("%d-%m-%Y %H:%M")))

    def train(self, dataset: Dataset, net_manager: NetManager):

        if os.path.exists(self.__log_filename):
            os.remove(self.__log_filename)
        self.__start_logger()

        # configure network
        logging.info('Initializing the net and compiling theano functions for the net...')
        logging.info('...Done')
        logging.info(str(net_manager.infos))

        return self._train(dataset, net_manager)

    def resume_training(self, dataset: Dataset, net_manager: NetManager):  # TODO load statistics too
        self.__start_logger()
        logging.info('Resuming training...')
        return self._train(dataset, net_manager)

    @staticmethod
    def compute_error_and_loss(loss_and_error_fnc, validation_batches: list):
        valid_error, valid_loss = 0., 0.
        for batch in validation_batches:
            valid_error_i, valid_loss_i = loss_and_error_fnc(batch.inputs, batch.outputs)
            valid_error += valid_error_i.item()
            valid_loss += valid_loss_i.item()
        n = len(validation_batches)
        return valid_error / n, valid_loss / n

    @staticmethod
    def __build_infos(train_info, i, valid_loss, valid_error, best_error, rho, batch_time, eval_time):

        it_info = PrintableInfoElement('iteration', ':7d', i)

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
