import theano
from SGDTrainer import SGDTrainer

from ActivationFunction import Tanh
from Configs import Configs
from ObjectiveFunction import ObjectiveFunction
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.ConstantInit import ConstantInit
from initialization.GaussianInit import GaussianInit
from initialization.SpectralInit import SpectralInit
from learningRule.GradientClipping import GradientClipping
from lossFunctions.FullSquaredError import FullSquaredError
from lossFunctions.SquaredError import SquaredError
from model.RNN import RNN
from model.RNNManager import RNNManager
from oldies.FixedAveragingOld import FixedAveragingOld
from output_fncs.Linear import Linear
from penalty.ConstantPenalty import ConstantPenalty
from datasets.AdditionTask import AdditionTask
from datasets.Dataset import InfiniteDataset
from datasets.PreTrainTask import PreTrainTask
from training.TrainingRule import TrainingRule

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
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
mean = 0
net_initializer = RNNManager(W_rec_init=SpectralInit(GaussianInit(mean=mean, std_dev=std_dev), rho=1.1), W_in_init=GaussianInit(mean=mean, std_dev = 0.1),
                             W_out_init=GaussianInit(mean=mean, std_dev=0.1), b_rec_init=ConstantInit(0),
                             b_out_init=ConstantInit(0), activation_fnc=Tanh(), output_fnc=Linear(), n_hidden=100)

# setup
seed = 13
orig_task = AdditionTask(144, seed)
pre_train_task = PreTrainTask(orig_task)

out_dir = Configs.output_dir + str(orig_task)+'_pretraining'

loss_fnc = FullSquaredError()

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
penalty = ConstantPenalty(c=5)
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
lr_rule = GradientClipping(lr_value=0.01, clip_thr=0.1)  # 0.01
# lr_rule = ArmijoStep(alpha=0.5, beta=0.1, init_step=1, max_steps=50)
obj_fnc = ObjectiveFunction(loss_fnc)

update_rule = FixedAveragingOld(t=10)
#update_rule = SimpleUdpate()
#update_rule = Momentum(gamma=0.1)

train_rule = TrainingRule(dir_rule, lr_rule, update_rule)

trainer = SGDTrainer(train_rule, obj_fnc, output_dir=out_dir, max_it=10000,
                     check_freq=200, batch_size=100)

# dataset = Dataset.no_valid_dataset_from_task(size=1000, datasets=datasets)
dataset = InfiniteDataset(task=pre_train_task, validation_size=10**4)
#net = trainer.train(dataset, net_initializer, seed)

net = RNN.load_model('/home/giulio/RNNs/models/add_task, min_length: 144_pretraining/current_model.npz')


# training
init = GaussianInit(mean=mean, std_dev=std_dev)
W_out = init.init_matrix((1, 100), dtype='float32')  # FIXME
b_out = init.init_matrix((1, 1), dtype='float32')  # FIXME
net = net.reconfigure_network(W_out, b_out, output_fnc=Linear())

obj_fnc = ObjectiveFunction(SquaredError())
out_dir = Configs.output_dir + str(orig_task)
dataset = InfiniteDataset(task=orig_task, validation_size=10**4)
trainer = SGDTrainer(train_rule, obj_fnc, output_dir=out_dir, max_it=10 ** 10,
                     check_freq=200, batch_size=100)
#net = trainer.train(dataset, net_initializer, seed)
# net = RNN.load_model(out_dir)
net = trainer.resume_training(dataset, net)
