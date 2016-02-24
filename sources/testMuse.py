import theano as T

from Paths import Paths
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from model.RNN import RNN
from task.MuseDataset import MuseDataset

__author__ = 'giulio'

separator = '#####################'

# setup
seed = 22
out_dir = '/home/giulio/RNNs/models/Muse/best_model.npz'

net = RNN.load_model(out_dir)

dataset = MuseDataset(seed=seed, pickle_file_path=Paths.muse_path, mode='full')


loss_fnc = FullCrossEntropy(single_probability_ouput=True)


loss = loss_fnc.value(t=net.symbols.t, y=net.symbols.y_shared)
loss_np = T.function([net.symbols.u, net.symbols.t, loss_fnc.mask], loss, name='loss_fnc')


loss = 0.
for b in dataset.test_set:
    loss += loss_np(b.inputs, b.outputs, b.mask).item()

print('Loss: {}'.format(loss/len(dataset.test_set)))


