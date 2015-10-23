from ActivationFunction import Tanh
from DescentDirectionRule import AntiGradientWithPenalty, MidAnglePenaltyDirection, FrozenGradient, AntiGradient
from LearningStepRule import ConstantNormalizedStep, ConstantStep, WRecNormalizedStep, ArmijoStep
from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from NetTrainer import NetTrainer
from TrainingRule import TrainingRule
from tasks.AdditionTask import AdditionTask
import theano
from Configs import Configs
from Penalty import MeanPenalty, NullPenalty, ConstantPenalty

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
dir_rule = AntiGradient()
#dir_rule = AntiGradientWithPenalty(penalty, 1) #0.001
#dir_rule = MidAnglePenaltyDirection(penalty)
#dir_rule = FrozenGradient(penalty)
#dir_rule = SepareteGradient()
#lr_rule = W_rec_step(0.0001) #0.01
lr_rule = ConstantStep(0.01) #0.01
#lr_rule = ConstantNormalizedStep(0.001) #0.01
#lr_rule = ArmijoStep(alpha=0.01, beta=0.1, init_step=0.0001, max_steps=50)
obj_fnc = ObjectiveFunction(loss_fnc, penalty, 14)
train_rule = TrainingRule(dir_rule, lr_rule)
trainer = NetTrainer(train_rule, obj_fnc)
net = trainer.train(task, activation_fnc, output_fnc, n_hidden, seed)

# train
net.train(train_rule)
