import datetime
import logging
import os
import time

import numpy
import theano as T
from numpy.linalg import LinAlgError

from Configs import Configs
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from metrics.Criterion import Criterion
from model import NetManager
from task.BatchPolicer import RepetitaPolicer
from task.Dataset import Dataset
from training.Statistics import Statistics
from training.TrainingRule import TrainingRule

__author__ = 'giulio'


class SGDTrainer(object):
    def __init__(self, training_rule: TrainingRule, stopping_criterion: Criterion, saving_criterion: Criterion,
                 monitors: list, max_it=10 ** 5, batch_size=100, check_freq=50, output_dir='.',
                 incremental_units: bool = False):

        # XXX for now the monitor used for the criterion must be passed in the monitor list

        self.__training_rule = training_rule
        self.__max_it = max_it
        self.__batch_size = batch_size
        self.__check_freq = check_freq
        self.__output_dir = output_dir
        self.__incremental_units = incremental_units

        self.__stopping_criterion = stopping_criterion
        self.__saving_criterion = saving_criterion
        self.__monitors = monitors

        self.__log_filename = self.__output_dir + '/train.log'

        # build training setting info
        self.__training_settings_info = InfoGroup('settings',
                                                  InfoList(PrintableInfoElement('max_it', ':d', self.__max_it),
                                                           PrintableInfoElement('check_freq', ':d', self.__check_freq),
                                                           PrintableInfoElement('batch_size', ':d', self.__batch_size),
                                                           self.__stopping_criterion.infos))

    def __compile_monitoring_fnc(self, net):

        u = net.symbols.u
        t = net.symbols.t
        y = net.symbols.y_shared  # XXX
        mask = self.__training_rule.loss_fnc.mask

        symbols = []
        symbols_lengths = []
        for m in self.__monitors:
            s = m.get_symbols(y=y, t=t)
            symbols += s
            symbols_lengths.append(len(s))

        return T.function([u, t, mask], symbols, name='monitoring_fnc'), symbols_lengths

    def __update_monitors(self, updated_values, symbols_lengths):
        start = 0
        for i in range(len(self.__monitors)):
            n = symbols_lengths[i]
            m = self.__monitors[i]
            values = updated_values[start:start + n]
            m.update(values)
            start += n

    def _train(self, dataset: Dataset, net_manager: NetManager, logger):

        net = net_manager.get_net(n_in=dataset.n_in, n_out=dataset.n_out)

        # add task description to infos
        self.__training_settings_info = InfoList(self.__training_settings_info,
                                                 PrintableInfoElement('task', '', str(dataset)))

        # compute_update symbols
        logger.info('Compiling theano functions for the training step...')
        train_step = self.__training_rule.compile(net)
        logger.info('... Done')
        logger.info('Seed: {}'.format(Configs.seed))

        monitor_fnc, symbols_lengths = self.__compile_monitoring_fnc(net)

        # training statistics
        stats = Statistics(self.__max_it, self.__check_freq, self.__training_settings_info, net.info)

        # policer #TODO se funziona mettere a pulito
        policer = RepetitaPolicer(dataset=dataset, batch_size=self.__batch_size, n_repetitions=100, block_size=1000)

        logger.info('Generating validation set ...')
        validation_set = dataset.validation_set
        logger.info('... Done')

        logger.info(str(net.info))

        logger.info(str(self.__training_settings_info))
        logger.info(str(self.__training_rule.infos))
        logger.info('Training ...\n')
        start_time = time.time()
        batch_start_time = time.time()

        error_occured = False
        i = 0

        while i < self.__max_it and not self.__stopping_criterion.is_satisfied() and not error_occured:

            # this makes the network grow in size according to the policy
            # specified when instanziating the network manager (may be even a null grow)
            net_manager.grow_net()

            batch = dataset.get_train_batch(self.__batch_size)
            # batch = policer.get_train_batch()

            if i % self.__check_freq == 0:  # FIXME 1st iteration

                train_info = train_step.step(batch.inputs, batch.outputs, batch.mask, report_info=True)
                eval_start_time = time.time()
                monitored_values = SGDTrainer.__evaluate_monitor_fnc(monitor_fnc, sum(symbols_lengths), validation_set)
                self.__update_monitors(monitored_values, symbols_lengths)

                try:
                    rho = net.spectral_radius
                except LinAlgError as e:
                    logger.error(str(e))
                    error_occured = True
                    rho = numpy.nan

                if not error_occured:

                    if self.__saving_criterion.is_satisfied():
                        logger.info('best model found: saving...')
                        net.save_model(self.__output_dir + '/best_model')

                    batch_end_time = time.time()
                    total_elapsed_time = batch_end_time - start_time

                    eval_time = time.time() - eval_start_time
                    batch_time = batch_end_time - batch_start_time

                    info = self.__build_infos(train_info, i, rho, batch_time, eval_time)
                    logger.info(info)
                    stats.update(info, i, total_elapsed_time)
                    net.save_model(self.__output_dir + '/current_model')
                    stats.save(self.__output_dir + '/stats')

                    batch_start_time = time.time()
                else:

                    logger.warning('stopping training...')
            else:
                train_step.step(batch.inputs, batch.outputs, batch.mask, report_info=False)

            i += 1

        end_time = time.time()
        if i == self.__max_it:
            logger.warning('Maximum number of iterations reached, stopping training...')
        elif self.__stopping_criterion.is_satisfied:
            logger.info('Training succeded, stopping criterion satisfied')
        logger.info('Elapsed time: {:2.2f} min'.format((end_time - start_time) / 60))
        return net

    def __start_logger(self):
        os.makedirs(self.__output_dir, exist_ok=True)

        file_handler = logging.FileHandler(filename=self.__log_filename, mode='a')
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger('rnn.train')  # root logger
        logger.setLevel(logging.INFO)

        for hdlr in logger.handlers:  # remove all old handlers
            logger.removeHandler(hdlr)
        logger.addHandler(file_handler)  # set the new handler
        now = datetime.datetime.now()
        logger.info('starting logging activity in date {}'.format(now.strftime("%d-%m-%Y %H:%M")))
        return logger

    def train(self, dataset: Dataset, net_manager: NetManager):

        if os.path.exists(self.__log_filename):
            os.remove(self.__log_filename)
        logger = self.__start_logger()

        # configure network
        logger.info('Initializing the net and compiling theano functions for the net...')
        logger.info('...Done')
        logger.info(str(net_manager.infos))

        return self._train(dataset, net_manager, logger)

    def resume_training(self, dataset: Dataset, net_manager: NetManager):  # TODO load statistics too
        logger = self.__start_logger()
        logger.info('Resuming training...')
        return self._train(dataset, net_manager, logger)

    @staticmethod
    def __evaluate_monitor_fnc(monitor_fnc, n, validation_batches: list):
        init_values = numpy.zeros(shape=(n,), dtype=Configs.floatType)
        for batch in validation_batches:
            values = monitor_fnc(batch.inputs, batch.outputs, batch.mask)
            assert (len(values) == n)
            for i in range(n):
                init_values[i] += values[i].item()
        m = len(validation_batches)
        return init_values / m

    def __build_infos(self, train_info, i, rho, batch_time, eval_time):

        it_info = PrintableInfoElement('iteration', ':7d', i)

        monitor_info = []
        for m in self.__monitors:
            monitor_info.append(m.info)

        val_info = InfoGroup('validation', InfoList(*monitor_info))

        rho_info = PrintableInfoElement('rho', ':5.2f', rho)
        eval_time_info = PrintableInfoElement('eval', ':2.2f', eval_time)
        batch_time_info = PrintableInfoElement('tot', ':2.2f', batch_time)
        time_info = InfoGroup('time', InfoList(eval_time_info, batch_time_info))

        info = InfoList(it_info, val_info, rho_info, train_info, time_info)
        return info
