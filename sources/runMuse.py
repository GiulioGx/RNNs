import theano

from ActivationFunction import Tanh
from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from SGDTrainer import SGDTrainer
from TrainingRule import TrainingRule
from combiningRule.EquiangularCombination import EquiangularCombination
from combiningRule.OnesCombination import OnesCombination
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.CrossEntropy import CrossEntropy
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from lossFunctions.SquaredError import SquaredError
from model import RNN
from model.RNNBuilder import RNNBuilder
from model.RNNInitializer import RNNInitializer
from output_fncs.Linear import Linear
from output_fncs.Logistic import Logistic
from output_fncs.Softmax import Softmax
from task.AdditionTask import AdditionTask
from task.Dataset import InfiniteDataset
from task.MuseDataset import MuseDataset
from task.TemporalOrderTask import TemporalOrderTask
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

# network setup
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
rnn_initializer = RNNInitializer(W_rec_init=GaussianInit(mean=mean, std_dev=std_dev, seed=seed),
                                 W_in_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed),
                                 W_out_init=GaussianInit(mean=mean, std_dev=0.1, seed=seed), b_rec_init=ConstantInit(0),
                                 b_out_init=ConstantInit(0))
net_builder = RNNBuilder(initializer=rnn_initializer, activation_fnc=Tanh(), output_fnc=Logistic(), n_hidden=100)

# setup
#task = TemporalOrderTask(144, seed)
out_dir = Configs.output_dir + 'Muse'
loss_fnc = FullCrossEntropy()

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

#combining_rule = OnesCombination(normalize_components=False)
combining_rule = SimplexCombination(normalize_components=True, seed=seed)
# combining_rule = SimpleSum()
#combining_rule = EquiangularCombination()
# combining_rule = DropoutCombination(drop_rate=0.8)
# combining_rule = MedianCombination()
dir_rule = CombinedGradients(combining_rule)
# dir_rule = DropoutDirection(dir_rule, drop_rate=0.1)
# dir_rule = DirectionWithPenalty(direction_rule=dir_rule, penalty=penalty, penalty_lambda=1)
# dir_rule = AlternatingDirections(dir_rule)

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
# lr_rule = ConstantNormalizedStep(0.001)  # 0.01
lr_rule = GradientClipping(lr_value=0.003, clip_thr=1, normalize_wrt_dimension=False)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)

#update_rule = FixedAveraging(t=10)
update_rule = SimpleUdpate()
# update_rule = Momentum(gamma=0.1)

train_rule = TrainingRule(dir_rule, lr_rule, update_rule, loss_fnc)

trainer = SGDTrainer(train_rule, output_dir=out_dir, max_it=10 ** 10,
                     check_freq=200, batch_size=100, stop_error_thresh=1)

# dataset = Dataset.no_valid_dataset_from_task(size=1000, task=task)
dataset = MuseDataset(seed=seed, pickle_file_path='/home/giulio/RNNs/datasets/polyphonic/musedata/MuseData.pickle')

net = trainer.train(dataset, net_builder, seed=seed)

#net = RNN.load_model(out_dir+'/best_model.npz')
#net = trainer.resume_training(dataset, net)
