from ActivationFunction import Tanh
from initialization.GaussianInit import GaussianInit
from initialization.ZeroInit import ZeroInit
from lossFunctions.CrossEntropy import CrossEntropy
from lossFunctions.HingeLoss import HingeLoss
from model import RNN
from task.XorTask import XorTask
import theano as T

seed = 15
print('Testing XOR task ...')
task = XorTask(22, seed)
batch = task.get_batch(3)
print(str(batch))

n_hidden = 50
activation_fnc = Tanh()
output_fnc = RNN.logistic
loss_fnc = CrossEntropy()
# init strategy
std_dev = 0.14  # 0.14 Tanh # 0.21 Relu
init_strategies = {'W_rec': GaussianInit(0, std_dev), 'W_in': GaussianInit(0, std_dev),
                   'W_out': GaussianInit(0, std_dev),
                   'b_rec': ZeroInit(), 'b_out': ZeroInit()}


net = RNN(activation_fnc, output_fnc, n_hidden, task.n_in, task.n_out, init_strategies, seed)


#  loss and error theano fnc
u = net.symbols.u
t = net.symbols.t
y = net.symbols.y_shared
error = task.error_fnc(y, t)
loss = loss_fnc.value(y, t)
loss_and_error = T.function([u, t], [error, loss, y], name='loss_and_error_fnc')


valid_error, valid_loss, y = loss_and_error(batch.inputs, batch.outputs)

print('y:', y[-1, :, :])
print(valid_error, valid_loss)
