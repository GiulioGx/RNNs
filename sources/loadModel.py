from Configs import Configs
from NetTrainer import NetTrainer
from ObjectiveFunction import ObjectiveFunction
from RNN import RNN
from TrainingRule import TrainingRule
from combiningRule.NormalizedSum import NormalizedSum
from combiningRule.SimpleSum import SimpleSum
from descentDirectionRule.AntiGradient import AntiGradient
from descentDirectionRule.CombinedGradients import CombinedGradients
from learningRule.ConstantNormalizedStep import ConstantNormalizedStep
from learningRule.ConstantStep import ConstantStep
from task.AdditionTask import AdditionTask

loadFile = '/home/giulio/RNNs/models/relu_34/model.npz'
log_filename = Configs.log_filename
model_filename = Configs.model_filename


net = RNN.load_model(loadFile)

task = AdditionTask(144, Configs.seed)

batch = task.get_batch(10)
y_net = net.net_output_shared(batch.inputs)  # FIXME


print(batch.outputs[-1, :, :])
print(y_net[-1, :, :])
print(((y_net[-1:, :, :] - batch.outputs[-1:, :, :]) ** 2).sum(axis=0))
print(task.error_fnc(batch.outputs, y_net))


loss_fnc = NetTrainer.squared_error
dir_rule = CombinedGradients()
combining_rule = NormalizedSum()
#lr_rule = ConstantStep(0.0004)  # 0.01
lr_rule = ConstantNormalizedStep(0.001)
obj_fnc = ObjectiveFunction(loss_fnc)
train_rule = TrainingRule(dir_rule, lr_rule, combining_rule)

trainer = NetTrainer(train_rule, obj_fnc, model_save_file=model_filename, log_filename=log_filename)

net = trainer.resume_training(task, net)
