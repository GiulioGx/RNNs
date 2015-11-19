import theano

from ActivationFunction import Tanh
from Configs import Configs
from NetTrainer import NetTrainer
from ObjectiveFunction import ObjectiveFunction
from TrainingRule import TrainingRule
from combiningRule.DropoutCombination import DropoutCombination
from combiningRule.EquiangularCombination import EquiangularCombination
from combiningRule.MedianCombination import MedianCombination
from combiningRule.NormalizedSum import NormalizedSum
from combiningRule.SimpleSum import SimpleSum
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.GaussianInit import GaussianInit
from initialization.ZeroInit import ZeroInit
from learningRule.ConstantNormalizedStep import ConstantNormalizedStep
from learningRule.ConstantStep import ConstantStep
from model.RNN import RNN
from penalty.NullPenalty import NullPenalty
from task.AdditionTask import AdditionTask
from learningRule.GradientClipping import GradientClipping

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

# setup
seed = 13
task = AdditionTask(144, seed)
n_hidden = 50
activation_fnc = Tanh()
output_fnc = RNN.linear_fnc
loss_fnc = NetTrainer.squared_error
out_dir = Configs.output_dir+'addition'

# init strategy
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
init_strategies = {'W_rec': GaussianInit(0, std_dev), 'W_in': GaussianInit(0, std_dev),
                   'W_out': GaussianInit(0, std_dev),
                   'b_rec': ZeroInit(), 'b_out': ZeroInit()}

# penalty strategy
# penalty = MeanPenalty()
# penalty = ConstantPenalty()
penalty = NullPenalty()

# direction strategy
# dir_rule = AntiGradient()
# dir_rule = AntiGradientWithPenalty(penalty, 1) #0.001
# dir_rule = MidAnglePenaltyDirection(penalty)
# dir_rule = FrozenGradient(penalty)
# dir_rule = SepareteGradient()

#combining_rule = SimplexCombination()
combining_rule = EquiangularCombination()
#combining_rule = DropoutCombination(drop_rate=0.8)
#combining_rule = MedianCombination()
dir_rule = CombinedGradients(combining_rule)

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
#lr_rule = ConstantStep(0.01)  # 0.01
lr_rule = GradientClipping(lr_value=0.01, clip_thr=0.1)  # 0.01
# lr_rule = ArmijoStep(alpha=0.1, beta=0.1, init_step=1, max_steps=50)

obj_fnc = ObjectiveFunction(loss_fnc, penalty, 0.1)
train_rule = TrainingRule(dir_rule, lr_rule)

trainer = NetTrainer(train_rule, obj_fnc, output_dir=out_dir, max_it=10 ** 10,
                     check_freq=200, bacth_size=20)

net = trainer.train(task, activation_fnc, output_fnc, n_hidden, init_strategies, seed)
