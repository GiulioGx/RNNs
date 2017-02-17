import datetime
import logging
import os
import time

import numpy
import theano as T
from numpy.linalg import LinAlgError

from Configs import Configs
from infos.Info import Info
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from metrics.Criterion import Criterion, AlwaysFalseCriterion, AlwaysTrueCriterion
from metrics.MeasureMonitor import MeasureMonitor
from model import NetManager
from datasets.BatchPolicer import RepetitaPolicer
from datasets.Dataset import Dataset
from training.Statistics import Statistics
from training.TrainingRule import TrainingRule

__author__ = 'giulio'


class SGDTrainer(object):
    def __init__(self, training_rule: TrainingRule, max_it=10 ** 5, batch_size=100, monitor_update_freq=50,
                 output_dir='.'):

        self.__training_rule = training_rule
        self.__max_it = max_it
        self.__batch_size = batch_size
        self.__check_freq = monitor_update_freq
        self.__output_dir = output_dir

        # XXX for now the monitors used for the criterions must be added manually

        # these are the deafult criterions, they can be changed with
        # the set_stopping_criterion and set_saving_criterion methods
        self.__stopping_criterion = AlwaysFalseCriterion()
        self.__saving_criterion = AlwaysFalseCriterion()

        self.__monitor_dicts = []

        self.__log_filename = self.__output_dir + '/train.log'

        # build training setting info
        self.__training_settings_info = InfoGroup('settings',
                                                  InfoList(PrintableInfoElement('max_it', ':d', self.__max_it),
                                                           PrintableInfoElement('check_freq', ':d', self.__check_freq),
                                                           PrintableInfoElement('batch_size', ':d', self.__batch_size)))

    def add_monitors(self, batches: list, name:str, *monitors: MeasureMonitor):
        self.__monitor_dicts.append(dict(batches=batches, monitors=monitors, name=name))

    def set_stopping_criterion(self, criterion: Criterion):
        self.__stopping_criterion = criterion

    def set_saving_criterion(self, criterion: Criterion):
        self.__saving_criterion = criterion

    def __compile_monitoring_fnc(self, net):

        u = net.symbols.u
        t = net.symbols.t
        y = net.symbols.y_shared  # XXX
        mask = net.symbols.mask

        for d in self.__monitor_dicts:

            monitors = d["monitors"]
            symbols = []
            symbols_lengths = []
            for m in monitors:
                s = m.get_symbols(y=y, t=t, mask=mask)
                symbols += s
                symbols_lengths.append(len(s))

            d["fnc"] = T.function([u, t, mask], symbols, name='monitoring_fnc')
            d["symbol_lengths"] = symbols_lengths

    def __update_monitors(self):
        for d in self.__monitor_dicts:
            lengths = d["symbol_lengths"]
            monitors = d["monitors"]
            batches = d["batches"]

            updated_values = []
            for batch in batches:
                updated_values.append(d["fnc"](batch.inputs, batch.outputs, batch.mask))

            start = 0
            for i in range(len(monitors)):
                monitor_values = []
                n = lengths[i]
                m = monitors[i]
                for batch_values in updated_values:
                    monitor_values.append(batch_values[start:start + n])
                start += n
                m.update(monitor_values)

    def _train(self, dataset: Dataset, net_manager: NetManager, logger):

        net = net_manager.get_net(n_in=dataset.n_in, n_out=dataset.n_out)

        # add datasets description to infos
        self.__training_settings_info = InfoList(self.__training_settings_info,
                                                 PrintableInfoElement('datasets', '', str(dataset)))

        # compute_update symbols
        logger.info('Compiling theano functions for the training step...')
        train_step = self.__training_rule.compile(net)
        logger.info('... Done')
        logger.info('Seed: {}'.format(Configs.seed))

        self.__compile_monitoring_fnc(net)

        # training statistics
        stats = Statistics(self.__max_it, self.__check_freq, self.__training_settings_info, net.info)

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
            net_manager.grow_net(logger)

            batch = dataset.get_train_batch(self.__batch_size)

            # W_rec = net.symbols.current_params.W_rec.get_value()
            # import numpy as np
            # print("std: {:.2f}, mean: {:.2f}".format(np.std(W_rec), np.mean(W_rec)))

            if i % self.__check_freq == 0:  # FIXME 1st iteration

                train_info, train_errors = train_step.step(batch.inputs, batch.outputs, batch.mask, report_info=True)
                SGDTrainer.__log_error(train_errors, logger)
                eval_start_time = time.time()
                self.__update_monitors()

                try:
                    spectral_info = net.spectral_info
                except LinAlgError as e:
                    logger.error(str(e))
                    error_occured = True

                if not error_occured:

                    if self.__saving_criterion.is_satisfied():
                        logger.info('best model found: saving...')
                        net.save_model(self.__output_dir + '/best_model')

                    batch_end_time = time.time()
                    total_elapsed_time = batch_end_time - start_time

                    eval_time = time.time() - eval_start_time
                    batch_time = batch_end_time - batch_start_time

                    info = self.__build_infos(train_info, i, spectral_info, batch_time, eval_time)
                    logger.info(info)
                    stats.update(info, i, total_elapsed_time)
                    net.save_model(self.__output_dir + '/current_model')
                    stats.save(self.__output_dir + '/stats')

                    batch_start_time = time.time()
                else:

                    logger.warning('stopping training...')
            else:
                train_errors = train_step.step(batch.inputs, batch.outputs, batch.mask, report_info=False)
                SGDTrainer.__log_error(train_errors, logger)

            i += 1

        end_time = time.time()
        if not error_occured:
            if i == self.__max_it:
                logger.warning('Maximum number of iterations reached, stopping training...')
            elif self.__stopping_criterion.is_satisfied:
                logger.info('Training succeded, stopping criterion satisfied')
            logger.info('Elapsed time: {:2.2f} min'.format((end_time - start_time) / 60))
        return net, stats

    def __start_logger(self):
        os.makedirs(self.__output_dir, exist_ok=True)

        file_handler = logging.FileHandler(filename=self.__log_filename, mode='a')
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        file_handler.setFormatter(formatter)

        logger = logging.getLogger('rnn.train'+self.__output_dir)
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

    def __build_infos(self, train_info, i, spectral_info, batch_time, eval_time):

        it_info = PrintableInfoElement('iteration', ':7d', i)

        monitors_info_list = []
        for d in self.__monitor_dicts:
            monitors = d["monitors"]
            monitor_info = []
            for m in monitors:
                monitor_info.append(m.info)
            monitors_info_list.append(InfoGroup(d["name"], InfoList(*monitor_info)))

        monitor_info = InfoList(*monitors_info_list)
        eval_time_info = PrintableInfoElement('eval', ':2.2f', eval_time)
        batch_time_info = PrintableInfoElement('tot', ':2.2f', batch_time)
        time_info = InfoGroup('time', InfoList(eval_time_info, batch_time_info))

        info = InfoList(it_info, monitor_info, spectral_info, train_info, time_info)
        return info

    @staticmethod
    def __log_error(errors:Info, logger):
        if errors.length > 0:
            logger.error(str(errors))
