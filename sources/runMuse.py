import theano

from ActivationFunction import Tanh
from Configs import Configs
from Paths import Paths
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.SpectralInit import SpectralInit
from initialization.UniformInit import UniformInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from metrics.BestValueFoundCriterion import BestValueFoundCriterion
from metrics.LossMonitor import LossMonitor
from metrics.ThresholdCriterion import ThresholdCriterion
from model.RNNInitializer import RNNInitializer, RNNVarsInitializer
from model.RNNManager import RNNManager
from output_fncs.Logistic import Logistic
from output_fncs.Softmax import Softmax
from task.MuseDataset import MuseDataset
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

seed = 13
Configs.seed = seed

# network setup
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
# vars_initializer = RNNVarsInitializer(
#     W_rec_init=SpectralInit(matrix_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed), rho=1.2),
#     W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
#     W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
#     b_out_init=ConstantInit(0))
vars_initializer = RNNVarsInitializer(
    W_rec_init=SpectralInit(matrix_init=GaussianInit(seed=seed, std_dev=0.14), rho=1.2),
    W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
    W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
    b_out_init=ConstantInit(0))
net_initializer = RNNInitializer(vars_initializer, n_hidden=100)
net_builder = RNNManager(initializer=net_initializer, activation_fnc=Tanh(), output_fnc=Logistic())

# setup
out_dir = Configs.output_dir + 'Muse'
loss_fnc = FullCrossEntropy(single_probability_ouput=True)

# # HF init
# bias_value = 0.5
# n_conns = 25
# std_dev = sqrt(0.12)
# init_strategies = {'W_rec': RandomConnectionsInit(n_connections_per_unit=n_conns, std_dev=std_dev, columnwise=False),
#                    'W_in': RandomConnectionsInit(n_connections_per_unit=n_conns, std_dev=0.1, columnwise=True),
#                    'W_out': RandomConnectionsInit(n_connections_per_unit=n_conns, std_dev=std_dev, columnwise=False),
#                    'b_rec': ConstantInit(bias_value), 'b_out': ConstantInit(bias_value)}

# penalty strategy
# penalty = MeanPenalty()
# penalty = ConstantPenalty(c=5)
# penalty = MeanPenalty()

# direction strategy
# dir_rule = AntiGradient()
# dir_rule = AntiGradientWithPenalty(penalty, 1) #0.001
# dir_rule = MidAnglePenaltyDirection(penalty)
# dir_rule = FrozenGradient(penalty)
# dir_rule = SepareteGradient()

# combining_rule = OnesCombination(normalize_components=False)
combining_rule = SimplexCombination(normalize_components=True, seed=seed)
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
lr_rule = GradientClipping(lr_value=0.01, clip_thr=1, normalize_wrt_dimension=False)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)

# update_rule = FixedAveraging(t=10)
update_rule = SimpleUdpate()
# update_rule = Momentum(gamma=0.1)

train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)

loss_monitor = LossMonitor(loss_fnc=loss_fnc)
monitors = [loss_monitor]
stopping_criterion = ThresholdCriterion(monitor=loss_monitor, threshold=0.9)
saving_criterion = BestValueFoundCriterion(monitor=loss_monitor)

trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                     check_freq=200, batch_size=100, saving_criterion=saving_criterion,
                     stopping_criterion=stopping_criterion, monitors=monitors)

dataset = MuseDataset(seed=seed, pickle_file_path=Paths.muse_path)

net = trainer.train(dataset, net_builder)

# net = RNN.load_model(out_dir+'/best_model.npz')
# net = trainer.resume_training(dataset, net)
