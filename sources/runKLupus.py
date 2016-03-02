import theano
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
from task.LupusDataset import LupusDataset
from training.SGDTrainer import SGDTrainer
from training.TrainingRule import TrainingRule
from updateRule.SimpleUpdate import SimpleUdpate

__author__ = 'giulio'


def thread_fnc(dataset, out_dir):
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
    roc_monitor = RocMonitor(score_fnc=LupusDataset.get_scores)
    stopping_criterion = ThresholdCriterion(monitor=roc_monitor, threshold=0.1, mode='>')
    saving_criterion = BestValueFoundCriterion(monitor=roc_monitor, mode='gt')

    trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                         monitor_update_freq=50, batch_size=20)
    trainer.add_monitors(dataset.train_set, 'train', loss_monitor, roc_monitor)
    trainer.set_saving_criterion(saving_criterion)
    trainer.set_stopping_criterion(stopping_criterion)

    net = trainer.train(dataset, net_builder)
    return net


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
k = 4

count = 0
for d in LupusDataset.k_fold_test_datasets(Paths.lupus_path, k=k):
    print('training net {}'.format(count))
    out_dir = Configs.output_dir + 'Lupus_k/' + str(count)
    net = thread_fnc(d, out_dir)
    count += 1
