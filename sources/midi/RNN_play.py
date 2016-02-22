import numpy

from Paths import Paths
from midi.MidiWriter import MidiWriter
from model import RNN
from task.MuseDataset import MuseDataset

seed = 676768
out_dir = '/home/giulio/RNNs/models/Bach/best_model.npz'

print('Loading...')
net = RNN.load_model(out_dir)
dataset = MuseDataset(seed=seed, pickle_file_path=Paths.bach_path, mode='full')
seq = dataset.test_set[5].inputs

print('RNN is composing music...')
played_music = net.net_ouput_numpy(seq)[0]
played_music = numpy.where(played_music > 0.5, 1, 0)

writer = MidiWriter()
writer.add_sequence(played_music)
writer.write_to_file("composed")

writer = MidiWriter()
writer.add_sequence(seq)
writer.write_to_file("original")
print('Ready to roll...')
