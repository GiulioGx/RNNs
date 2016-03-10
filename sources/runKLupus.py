import logging
import os
import pickle
from threading import Thread

import theano
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from ActivationFunction import Tanh
from Configs import Configs
from Paths import Paths
from descentDirectionRule.Antigradient import Antigradient
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.SpectralInit import SpectralInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from metrics.BestValueFoundCriterion import BestValueFoundCriterion
from metrics.LossMonitor import LossMonitor
from metrics.RocMonitor import RocMonitor
from metrics.ThresholdCriterion import ThresholdCriterion
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer
from model.RNNManager import RNNManager
from output_fncs.Logistic import Logistic
from task.LupusDataset import LupusDataset, PerPatienceTargets
from training.SGDTrainer import SGDTrainer
from training.TrainingRule import TrainingRule
from updateRule.SimpleUpdate import SimpleUdpate

__author__ = 'giulio'


class SplitThread(Thread):
    def __init__(self, out_dir: str, dataset, id: int, logger):
        super().__init__()
        self.__net = None
        self.__dataset = dataset
        self.__out_dir = out_dir
        self.__id = id
        self.__results = None
        self.__logger = logger

    @property
    def net(self):
        return self.__net

    @property
    def results(self):
        return self.__results

    def __train(self):
        self.__logger.info('Starting thread {}'.format(self.__id))
        # network setup
        std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
        mean = 0
        vars_initializer = RNNVarsInitializer(
            W_rec_init=SpectralInit(matrix_init=GaussianInit(seed=seed, std_dev=std_dev), rho=1.2),
            W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
            W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
            b_out_init=ConstantInit(0))
        net_initializer = RNNInitializer(vars_initializer, n_hidden=50)
        net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(),
                                 output_fnc=Logistic())

        loss_fnc = FullCrossEntropy(single_probability_ouput=True)
        dir_rule = Antigradient()
        lr_rule = GradientClipping(lr_value=0.001, clip_thr=1, normalize_wrt_dimension=False)  # 0.01
        update_rule = SimpleUdpate()
        train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc, nan_check=True)

        loss_monitor = LossMonitor(loss_fnc=loss_fnc)
        roc_monitor = RocMonitor(score_fnc=LupusDataset.get_scores_patients)
        stopping_criterion = ThresholdCriterion(monitor=roc_monitor, threshold=0.1, mode='>')
        saving_criterion = BestValueFoundCriterion(monitor=roc_monitor, mode='gt')

        trainer = SGDTrainer(train_rule, output_dir=self.__out_dir, max_it=10 ** 10,
                             monitor_update_freq=50, batch_size=20)
        trainer.add_monitors(self.__dataset.train_set, 'train', loss_monitor, roc_monitor)
        trainer.set_saving_criterion(saving_criterion)
        trainer.set_stopping_criterion(stopping_criterion)

        self.__net = trainer.train(self.__dataset, net_builder)

    def __test(self):
        batch = self.__dataset.split_test['late_pos']
        y = self.__net.net_ouput_numpy(batch.inputs)[0]
        scores, labels = LupusDataset.get_scores_patients(y, batch.outputs, batch.mask)
        late_fpr, late_tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)

        batch = self.__dataset.test_set[0]
        y = self.__net.net_ouput_numpy(batch.inputs)[0]
        scores, labels = LupusDataset.get_scores_patients(y, batch.outputs, batch.mask)
        # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
        score = roc_auc_score(y_true=labels, y_score=scores)

        result = dict(score=score, late_pos_fpr=late_fpr, late_pos_tpr=late_tpr)
        return result

    def run(self):
        self.__train()
        # save dataset to disk
        pickfile = open(self.__out_dir + '/dataset.pkl', "wb")
        pickle.dump(d, pickfile)
        self.__logger.info('Thread {} has fineshed training -> beginning test'.format(self.__id))
        self.__results = self.__test()
        self.__logger.info('Partial results for thread {} are:\n {}'.format(self.__id, self.__results))
        self.__logger.info('Thread {} has finished....'.format(self.__id))


separator = '#####################'

# ###THEANO CONFIG ### #
floatX = theano.config.floatX
device = theano.config.device
Configs.floatType = floatX
print(separator)
print('THEANO CONFIG')
print('device: ' + device)
print('floatType: ' + floatX)
print(separator)

seed = 15
Configs.seed = seed
k = 8

# start main logger
root_out_dir = Configs.output_dir + 'Lupus_k/'
os.makedirs(root_out_dir, exist_ok=True)
log_filename = root_out_dir + 'train_k.log'
if os.path.exists(log_filename):
    os.remove(log_filename)
file_handler = logging.FileHandler(filename=log_filename, mode='a')
formatter = logging.Formatter('%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
logger = logging.getLogger('split.train')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.info('Begin {}-split training'.format(k))

count = 0
thread_list = []
for d in LupusDataset.k_fold_test_datasets(Paths.lupus_path, k=k, strategy=PerPatienceTargets()):
    out_dir = root_out_dir + str(count)
    t = SplitThread(out_dir=out_dir, dataset=d, id=count, logger=logger)
    thread_list.append(t)
    t.start()
    count += 1

score = 0.
# Wait for all threads to complete
for t in thread_list:
    t.join()
    score += t.results['score']

logger.info("All thread finished...")
logger.info('ROC_AUC score: {:.2f}'.format(score / len(thread_list)))
