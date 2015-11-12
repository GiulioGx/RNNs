from Configs import Configs
from NetTrainer import NetTrainer
from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from TrainingRule import TrainingRule
from combiningRule.NormalizedSum import NormalizedSum
from combiningRule.SimpleSum import SimpleSum
from combiningRule.SimplexCombination import SimplexCombination
from descentDirectionRule.AntiGradient import AntiGradient
from descentDirectionRule.CombinedGradients import CombinedGradients
from learningRule.ConstantNormalizedStep import ConstantNormalizedStep
from learningRule.ConstantStep import ConstantStep
from task.AdditionTask import AdditionTask

loadFile = '/home/giulio/RNNs/models/model_add_p.npz'
log_filename = Configs.log_filename+'_cont'
model_filename = Configs.output_dir + '_cont'

Configs.seed = 23


net = RNN.load_model(loadFile)

task = AdditionTask(144, Configs.seed)


loss_fnc = NetTrainer.squared_error
combining_rule = SimplexCombination()
dir_rule = CombinedGradients(combining_rule)
#lr_rule = ConstantStep(0.0004)  # 0.01
lr_rule = ConstantNormalizedStep(0.001) #0.01
obj_fnc = ObjectiveFunction(loss_fnc)
train_rule = TrainingRule(dir_rule, lr_rule)

trainer = NetTrainer(train_rule, obj_fnc, model_save_file=model_filename, log_filename=log_filename, max_it=10**5, check_freq=100)

net = trainer.resume_training(task, net)
