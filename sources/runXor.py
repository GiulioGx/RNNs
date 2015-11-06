import theano
from ActivationFunction import Tanh, Relu
from combiningRule.NormalizedSum import NormalizedSum
from combiningRule.SimpleSum import SimpleSum
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.AntiGradient import AntiGradient
from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from NetTrainer import NetTrainer
from TrainingRule import TrainingRule
from descentDirectionRule.CombinedGradients import CombinedGradients
from initialization.GaussianInit import GaussianInit
from initialization.UniformInit import UniformInit
from initialization.ZeroInit import ZeroInit
from learningRule.ArmijoStep import ArmijoStep
from learningRule.ConstantNormalizedStep import ConstantNormalizedStep
from learningRule.ConstantStep import ConstantStep
from penalty.ConstantPenalty import ConstantPenalty
from task.AdditionTask import AdditionTask
from Configs import Configs
from penalty.NullPenalty import NullPenalty
from task.XorTask import XorTask

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
task = XorTask(144, seed)
n_hidden = 50
activation_fnc = Tanh()
output_fnc = RNN.linear_fnc
loss_fnc = NetTrainer.squared_error
model_filename = Configs.model_filename+'_xor'
log_filename = Configs.log_filename+'_xor'

# init strategy
std_dev = 0.14  # 0.14 Tanh #0.21 Relu
init_strategies = {'W_rec': GaussianInit(0, std_dev), 'W_in': GaussianInit(0, std_dev),
                   'W_out': GaussianInit(0, std_dev),
                   'b_rec': ZeroInit(), 'b_out': ZeroInit()}


# direction strategy
# dir_rule = AntiGradient()
# dir_rule = AntiGradientWithPenalty(penalty, 1) #0.001
# dir_rule = MidAnglePenaltyDirection(penalty)
# dir_rule = FrozenGradient(penalty)
# dir_rule = SepareteGradient()
dir_rule = CombinedGradients()
combining_rule = NormalizedSum()

# learning step rule
# lr_rule = WRecNormalizedStep(0.0001) #0.01
#lr_rule = ConstantStep(0.0004)  # 0.01
lr_rule = ConstantNormalizedStep(0.0001) #0.01
# lr_rule = ArmijoStep(alpha=0.1, beta=0.1, init_step=1, max_steps=50)

obj_fnc = ObjectiveFunction(loss_fnc)
train_rule = TrainingRule(dir_rule, lr_rule, combining_rule)

trainer = NetTrainer(train_rule, obj_fnc, model_save_file=model_filename, log_filename=log_filename)

net = trainer.train(task, activation_fnc, output_fnc, n_hidden, init_strategies, seed)
