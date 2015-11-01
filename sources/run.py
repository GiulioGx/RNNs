import theano

from ActivationFunction import Tanh, Relu
from combiningRule.NormalizedSum import NormalizedSum
from combiningRule.SimpleSum import SimpleSum
from combiningRule.StochasticCombination import StochasticCombination
from descentDirectionRule.AntiGradient import AntiGradient
from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from NetTrainer import NetTrainer
from TrainingRule import TrainingRule
from descentDirectionRule.CombinedGradients import CombinedGradients
from learningRule.ArmijoStep import ArmijoStep
from learningRule.ConstantNormalizedStep import ConstantNormalizedStep
from learningRule.ConstantStep import ConstantStep
from penalty.ConstantPenalty import ConstantPenalty
from task.AdditionTask import AdditionTask
from Configs import Configs
from penalty.NullPenalty import NullPenalty

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
output_fnc = RNN.last_linear_fnc
loss_fnc = NetTrainer.squared_error
#penalty = MeanPenalty()
#penalty = ConstantPenalty()
penalty = NullPenalty()
#dir_rule = AntiGradient()
#dir_rule = AntiGradientWithPenalty(penalty, 1) #0.001
#dir_rule = MidAnglePenaltyDirection(penalty)
#dir_rule = FrozenGradient(penalty)
#dir_rule = SepareteGradient()
dir_rule = CombinedGradients()
combining_rule = NormalizedSum()
#lr_rule = WRecNormalizedStep(0.0001) #0.01
lr_rule = ConstantStep(0.001) #0.01
#lr_rule = ConstantNormalizedStep(0.001) #0.01
#lr_rule = ArmijoStep(alpha=0.1, beta=0.1, init_step=1, max_steps=50)
obj_fnc = ObjectiveFunction(loss_fnc, penalty, 0.1)
train_rule = TrainingRule(dir_rule, lr_rule, combining_rule)
trainer = NetTrainer(train_rule, obj_fnc)


net = trainer.train(task, activation_fnc, output_fnc, n_hidden, seed)

