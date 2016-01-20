import theano
from math import sqrt

from ActivationFunction import Tanh, Relu
from Configs import Configs
from SGDTrainer import SGDTrainer
from ObjectiveFunction import ObjectiveFunction
from TrainingRule import TrainingRule
from descentDirectionRule.DropoutDirection import DropoutDirection
from initialization.ConstantInit import ConstantInit
from initialization.EyeInit import EyeInit
from initialization.OrtoghonalInit import OrtoghonalInit
from initialization.SparseGaussianInit import SparseGaussianInit
from initialization.SimplexInit import SimplexInit
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from lossFunctions.CrossEntropy import CrossEntropy
from lossFunctions.HingeLoss import HingeLoss
from lossFunctions.NullLoss import NullLoss
from lossFunctions.SquaredError import SquaredError
from model.RNNBuilder import RNNBuilder
from output_fncs.Softmax import Softmax
from output_fncs.Linear import Linear
from task.Dataset import Dataset, InfiniteDataset
from task.TemporalOrderTask import TemporalOrderTask
from task.XorTask import XorTask
from task.XorTaskHot import XorTaskHot
from updateRule.FixedAveraging import FixedAveraging
from updateRule.Momentum import Momentum
from updateRule.SimpleUpdate import SimpleUdpate
from combiningRule.DropoutCombination import DropoutCombination
from combiningRule.EquiangularCombination import EquiangularCombination
from combiningRule.MedianCombination import MedianCombination
from combiningRule.OnesCombination import OnesCombination
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.AlternatingDirections import AlternatingDirections
from descentDirectionRule.CombinedGradients import CombinedGradients
from descentDirectionRule.DirectionWithPenalty import DirectionWithPenalty
from initialization.GaussianInit import GaussianInit
from initialization.ZeroInit import ZeroInit
from learningRule.ArmijoStep import ArmijoStep
from learningRule.ConstantNormalizedStep import ConstantNormalizedStep
from learningRule.ConstantStep import ConstantStep
from model.RNN import RNN
from penalty.ConstantPenalty import ConstantPenalty
from penalty.MeanPenalty import MeanPenalty
from penalty.NullPenalty import NullPenalty
from task.AdditionTask import AdditionTask
from learningRule.GradientClipping import GradientClipping
from task.MultiplicationTask import MultiplicationTask

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

# network setup
# std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
# mean = 0
# net_initializer = RnnInitializer(W_rec_init=GaussianInit(mean=mean, std_dev=std_dev), W_in_init=GaussianInit(mean=mean, std_dev = 0.1),
#                                  W_out_init=GaussianInit(mean=mean, std_dev=0.1), b_rec_init=ConstantInit(0),
#                                  b_out_init=ConstantInit(0), activation_fnc=Tanh(), output_fnc=Linear(), n_hidden=100)

# setup
seed = 13
task = AdditionTask(144, seed)
out_dir = Configs.output_dir + str(task)
loss_fnc = SquaredError()

# # HF init
bias_value = 0
n_conns = 15
std_dev = 0.1
#std_dev = 0.14
net_initializer = RNNBuilder(
    W_rec_init=SpectralInit(SparseGaussianInit(n_connections_per_unit=n_conns, std_dev=std_dev, columnwise=False), rho=1.1),
    W_in_init=SparseGaussianInit(n_connections_per_unit=n_conns, std_dev=1, columnwise=True),
    W_out_init=SparseGaussianInit(n_connections_per_unit=n_conns, std_dev=std_dev, columnwise=False),
    b_rec_init=ConstantInit(0),
    b_out_init=ConstantInit(0), activation_fnc=Tanh(), output_fnc=Linear(), n_hidden=100)

# penalty strategy
# penalty = MeanPenalty()
#penalty = ConstantPenalty(c=5)
# penalty = MeanPenalty()

# direction strategy
# dir_rule = AntiGradient()
# dir_rule = AntiGradientWithPenalty(penalty, 1) #0.001
# dir_rule = MidAnglePenaltyDirection(penalty)
# dir_rule = FrozenGradient(penalty)
# dir_rule = SepareteGradient()

# combining_rule = OnesCombination(normalize_components=False)
combining_rule = SimplexCombination(normalize_components=True)
# combining_rule = SimpleSum()
# combining_rule = EquiangularCombination()
# combining_rule = DropoutCombination(drop_rate=0.8)
# combining_rule = MedianCombination()
dir_rule = CombinedGradients(combining_rule)
# dir_rule = DropoutDirection(dir_rule, drop_rate=0.1)
# dir_rule = DirectionWithPenalty(direction_rule=dir_rule, penalty=penalty, penalty_lambda=1)
# dir_rule = AlternatingDirections(dir_rule)

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
# lr_rule = ConstantNormalizedStep(0.001)  # 0.01
lr_rule = GradientClipping(lr_value=0.001, clip_thr=0.1)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)
obj_fnc = ObjectiveFunction(loss_fnc)

update_rule = FixedAveraging(t=10)
# update_rule = SimpleUdpate()
#update_rule = Momentum(gamma=0.1)

train_rule = TrainingRule(dir_rule, lr_rule, update_rule)

trainer = SGDTrainer(train_rule, obj_fnc, output_dir=out_dir, max_it=10 ** 10,
                     check_freq=200, batch_size=100, stop_error_thresh=0.1)

# dataset = Dataset.no_valid_dataset_from_task(size=1000, task=task)
dataset = InfiniteDataset(task=task, validation_size=10 ** 4)

net = trainer.train(dataset, net_initializer, seed)

# net = RNN.load_model(out_dir+'/current_model.npz')
# net = trainer.resume_training(dataset, net)
