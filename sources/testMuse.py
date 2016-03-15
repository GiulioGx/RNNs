import theano as T

from Paths import Paths
from lossFunctions.FullCrossEntropy import FullCrossEntropy
from model.RNN import RNN
from datasets.MuseDataset import MuseDataset

__author__ = 'giulio'

separator = '#####################'

# setup
seed = 22
out_dir = '/home/giulio/MuseCollection/0/best_model.npz'

net = RNN.load_model(out_dir)

dataset = MuseDataset(seed=seed, pickle_file_path=Paths.muse_path, mode='full')


loss_fnc = FullCrossEntropy(single_probability_ouput=True)


loss = loss_fnc.value(t=net.symbols.t, y=net.symbols.y_shared, mask=net.symbols.mask)
loss_np = T.function([net.symbols.u, net.symbols.t, net.symbols.mask], loss, name='loss_fnc')


loss = 0.
for b in dataset.test_set:
    loss += loss_np(b.inputs, b.outputs, b.mask).item()
print('Test loss: {}'.format(loss/len(dataset.test_set)))

loss = 0.
for b in dataset.train_set:
    loss += loss_np(b.inputs, b.outputs, b.mask).item()
print('Train loss: {}'.format(loss/len(dataset.train_set)))




