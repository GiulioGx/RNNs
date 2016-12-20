import theano, numpy
from neuralflow.enel.enel import export_results

from ActivationFunction import Tanh
from Configs import Configs
from descentDirectionRule.Antigradient import Antigradient
from descentDirectionRule.CheckedDirection import CheckedDirection
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.SVDInit import SVDInit
from initialization.SparseGaussianInit import SparseGaussianInit
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from metrics.BestValueFoundCriterion import BestValueFoundCriterion
from metrics.ErrorMonitor import ErrorMonitor
from metrics.LossMonitor import LossMonitor
from metrics.ThresholdCriterion import ThresholdCriterion
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer
from model.RNNManager import RNNManager
from output_fncs.Softmax import Softmax
from datasets.Dataset import InfiniteDataset
from datasets.TemporalOrderTask import TemporalOrderTask
from training.SGDTrainer import SGDTrainer
from training.TrainingRule import TrainingRule
from updateRule.SimpleUpdate import SimpleUdpate

__author__ = 'giulio'

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


def run(parameters, task, output_dir, id):
    # network setup
    std_dev = parameters["std_dev"]  # 0.14 Tanh # 0.21 Relu
    mean = 0
    vars_initializer = RNNVarsInitializer(
        W_rec_init=SpectralInit(matrix_init=GaussianInit(mean=mean, std_dev=.01, seed=seed), rho=1.2),
        W_in_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed),
        W_out_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed), b_rec_init=ConstantInit(0),
        b_out_init=ConstantInit(0))
    # vars_initializer = RNNVarsInitializer(
    #     W_rec_init=GaussianInit(seed=seed, std_dev=std_dev),
    #     W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
    #     W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
    #     b_out_init=ConstantInit(0))
    net_initializer = RNNInitializer(vars_initializer, n_hidden=50)
    net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(),
                             output_fnc=Softmax())

    # setup
    loss_fnc = FullCrossEntropy(single_probability_ouput=False)

    # combining_rule = SimpleSum()
    # dir_rule = CombinedGradients(combining_rule)
    # dir_rule = CheckedDirection(dir_rule, max_cos=0, max_dir_norm=numpy.inf)
    dir_rule = Antigradient()

    # learning step rule
    lr_rule = GradientClipping(lr_value=parameters["lr"], clip_thr=parameters["clip_thr"], clip_style='l1')  # 0.01
    update_rule = SimpleUdpate()

    train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)

    dataset = InfiniteDataset(task=task, validation_size=10 ** 4, n_batches=5)
    val_batches = dataset.validation_set
    print("validation set batches:")
    for b in val_batches:
        print("\t", b.inputs.shape)

    loss_monitor = LossMonitor(loss_fnc=loss_fnc)
    error_monitor = ErrorMonitor(dataset=dataset, error_fnc=task.error_fnc)
    stopping_criterion = ThresholdCriterion(monitor=error_monitor, threshold=1. / 100)
    saving_criterion = BestValueFoundCriterion(monitor=error_monitor)

    trainer = SGDTrainer(train_rule, output_dir=output_dir + "{}/".format(id), max_it=2000000,
                         monitor_update_freq=200, batch_size=20)  # update_freq 200
    trainer.add_monitors(dataset.validation_set, "validation", loss_monitor, error_monitor)
    trainer.set_saving_criterion(saving_criterion)
    trainer.set_stopping_criterion(stopping_criterion)

    net, stats = trainer.train(dataset, net_builder)
    n_iters = stats.n_iters
    return n_iters


if __name__ == "__main__":

    seed = 14
    Configs.seed = seed

    task = TemporalOrderTask(70, seed)
    output_dir = Configs.output_dir + str(task) + '_' + str(seed) + "_multi_run/"

    id = 0
    result_list = []
    for lr in numpy.linspace(0.01, 0.1, 10):
        for clip_thr in numpy.linspace(0.01, 0.1, 10):
            for std_dev in [0.01, 0.05, 0.1, 0.15]:
                parameters = {
                    "id": id,
                    "lr": lr,
                    "clip_thr": clip_thr,
                    "std_dev": std_dev,
                }
                print("Beginning instance {}...".format(id))
                n_iters = run(parameters, task, output_dir, id)
                parameters.update({
                    "n_iters": n_iters})
                result_list.append(parameters)
                export_results(result_list, output_dir)
                id += 1
