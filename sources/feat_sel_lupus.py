import logging
import os
import pickle
from threading import Thread

import numpy
import shutil
import theano
from sklearn import metrics
from sklearn.metrics import roc_auc_score

from ActivationFunction import Tanh
from Configs import Configs
from Paths import Paths
from datasets.LupusFilter import TemporalSpanFilter
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
from datasets.LupusDataset import LupusDataset, PerPatienceTargets, TemporalDifferenceTargets, LastAndFirstVisitsTargets
from training.SGDTrainer import SGDTrainer
from training.TrainingRule import TrainingRule
from updateRule.SimpleUpdate import SimpleUdpate

__author__ = 'giulio'


class SplitThread(Thread):
    def __init__(self, out_dir: str, dataset, id: int, logger, n_hidden: int, stop_thr: float):
        super().__init__()
        self.__net = None
        self.__dataset = dataset
        self.__out_dir = out_dir
        self.__id = id
        self.__results = None
        self.__logger = logger
        self.__n_hidden = n_hidden
        self.__stop_thr = stop_thr

    @property
    def net(self):
        return self.__net

    @property
    def results(self):
        return self.__results

    def __train(self):
        self.__logger.info('Starting thread {}'.format(self.__id))
        # network setup
        std_dev = 0.1  # 0.14 Tanh # 0.21 Relu
        mean = 0
        vars_initializer = RNNVarsInitializer(
            W_rec_init=SpectralInit(matrix_init=GaussianInit(seed=seed, std_dev=std_dev), rho=1.2),
            W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
            W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
            b_out_init=ConstantInit(0))
        net_initializer = RNNInitializer(vars_initializer, n_hidden=self.__n_hidden)
        net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(),
                                 output_fnc=Logistic())

        loss_fnc = FullCrossEntropy(single_probability_ouput=True)
        dir_rule = Antigradient()
        lr_rule = GradientClipping(lr_value=0.001, clip_thr=1)  # 0.01
        update_rule = SimpleUdpate()
        train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc, nan_check=True)

        loss_monitor = LossMonitor(loss_fnc=loss_fnc)
        roc_monitor = RocMonitor(score_fnc=LupusDataset.get_scores_patients)
        stopping_criterion = ThresholdCriterion(monitor=roc_monitor, threshold=self.__stop_thr, mode='>')
        saving_criterion = BestValueFoundCriterion(monitor=roc_monitor, mode='gt')

        trainer = SGDTrainer(train_rule, output_dir=self.__out_dir, max_it=50000,
                             monitor_update_freq=50, batch_size=20)
        trainer.add_monitors(self.__dataset.train_set, 'train', loss_monitor, roc_monitor)
        trainer.set_saving_criterion(saving_criterion)
        trainer.set_stopping_criterion(stopping_criterion)

        self.__net, _ = trainer.train(self.__dataset, net_builder)

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

        result = dict(score=score, late_pos_fpr=late_fpr, late_pos_tpr=late_tpr, y=y, batch=batch)
        return result

    def run(self):
        self.__train()
        # save dataset to disk
        pickfile = open(self.__out_dir + '/dataset.pkl', "wb")
        pickle.dump(self.__dataset, pickfile)
        self.__logger.info('Thread {} has fineshed training -> beginning test'.format(self.__id))
        self.__results = self.__test()
        self.__logger.info('Partial score for thread {} is: {:.2f}'.format(self.__id, self.__results['score']))
        self.__logger.info('Thread {} has finished....'.format(self.__id))


def run_experiment(root_dir, min_age_lower, min_age_upper, min_visits_neg, min_visits_pos, n_hidden, stop_thr, id: int,
                   feats):
    # start main logger
    run_out_dir = root_dir + 'run_{}/'.format(id)
    os.makedirs(run_out_dir, exist_ok=True)
    log_filename = run_out_dir + 'train_k.log'
    if os.path.exists(log_filename):
        os.remove(log_filename)
    file_handler = logging.FileHandler(filename=log_filename, mode='a')
    formatter = logging.Formatter('%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('split.train_{}'.format(id))
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info('Begin {}-split training'.format(k))

    thread_count = 0
    thread_list = []
    dataset_infos = None
    strategy = PerPatienceTargets()
    # strategy = LastAndFirstVisitsTargets()
    for d in LupusDataset.k_fold_test_datasets(Paths.lupus_path, k=k, strategy=strategy,
                                               visit_selector=TemporalSpanFilter(
                                                   min_age_span_upper=min_age_upper,
                                                   min_age_span_lower=min_age_lower, min_visits_neg=min_visits_neg,
                                                   min_visits_pos=min_visits_pos), feats=feats):
        out_dir = run_out_dir + str(thread_count)
        t = SplitThread(out_dir=out_dir, dataset=d, id=thread_count, logger=logger, n_hidden=n_hidden,
                        stop_thr=stop_thr)
        thread_list.append(t)
        t.start()
        t.join()
        thread_count += 1
        dataset_infos = d.infos

    ys = []
    masks = []
    outputs = []
    score = 0.
    # Wait for all threads to complete
    for t in thread_list:
        # sudo t.join()
        score += t.results['score']
        ys.append(t.results['y'])
        batch = t.results['batch']
        masks.append(batch.mask)
        outputs.append(batch.outputs)

    y = numpy.concatenate(ys, axis=2)
    mask = numpy.concatenate(masks, axis=2)
    output = numpy.concatenate(outputs, axis=2)
    scores, labels = LupusDataset.get_scores_patients(y, output, mask)
    # fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    cum_score = roc_auc_score(y_true=labels, y_score=scores)

    logger.info("All thread finished...")
    logger.info('ROC_AUC mean score: {:.2f}'.format(score / len(thread_list)))
    logger.info('ROC_AUC cumulative score: {:.2f}'.format(cum_score))

    # save scores to file
    npz_file = run_out_dir + 'scores.npz'
    os.makedirs(os.path.dirname(npz_file), exist_ok=True)
    save_info = dict(scores=scores, labels=labels, cum_score=cum_score, mean_score=score)
    save_info.update(dataset_infos.dictionary)
    numpy.savez(npz_file, **save_info)

    return cum_score


if __name__ == '__main__':

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

    min_age_lower = 0.8  # 0.8, 1, 2]
    min_age_upper = 0.8  # [0.8, 1, 2]
    min_num_visits_neg = 1  # [1, 2, 3, 4, 5] #5
    min_num_visits_pos = 1
    n_hidden = 100
    stop_thr = 0.95

    feats = ['APS', 'DNA', 'FM', 'Hashimoto', 'MyasteniaGravis', 'SdS',
             'arterialthrombosis', 'arthritis', 'c3level', 'c4level', 'dislipidemia', 'hcv',
             'hematological', 'hypertension', 'hypothyroidism', 'kidney', 'mthfr', 'npsle',
             'pregnancypathology', 'serositis', 'sex', 'skinrash', 'sledai2kInferred',
             'venousthrombosis']

    root_dir = Configs.output_dir + 'Lupus_feat_sel_thr/'
    shutil.rmtree(root_dir, ignore_errors=True)

    count = 0
    j = 0
    best_score = 0
    trail_feats = feats
    rnd = numpy.random.RandomState(seed)

    score_thr = 0.01

    exit = False
    while not exit:

        print("Executing experiment with {} feats: {}".format(len(trail_feats), str(trail_feats)))

        score = run_experiment(root_dir=root_dir, min_age_lower=min_age_lower, min_age_upper=min_age_upper,
                               min_visits_neg=min_num_visits_neg, min_visits_pos=min_num_visits_pos, id=count,
                               n_hidden=n_hidden,
                               stop_thr=stop_thr, feats=trail_feats)

        if score >= best_score - score_thr:
            print("Best score found: {:.2f}".format(score))
            best_score = score
            j = 0
            feats = trail_feats
            rnd.shuffle(feats)

        else:
            print("Got score {:.2f} which is not an improvement of the best score: {:.2f}.".format(score, best_score))
            if j == len(feats) - 1:
                exit = True
                print("Quitting->an unsuccesful loop was performed.")
            else:
                j += 1

        if len(feats) == 1:
            exit = True
            print("Quitting-> Min subset (1 feats) found.")
        else:
            print("Next feature to be tried: {}; j={}.".format(feats[j], j))
            trail_feats = feats[:j] + feats[j + 1:]

        count += 1

print("Best feat subset: {}".format(str(feats)))
