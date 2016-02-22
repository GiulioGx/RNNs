import numpy
from midiutil.MidiFile3 import MIDIFile
from Paths import Paths
from task.MuseDataset import MuseDataset


class MidiWriter(object):

    def __init__(self):
        self.__midi_obj = MIDIFile(1)
        # Tracks are numbered from zero. Times are measured in beats.
        self.__track = 0
        self.__channel = 1
        self.__duration = 1
        self.__volume = 100
        time = 0
        # Add track name and tempo.
        self.__midi_obj.addTrackName(self.__track, time, "RNN playing Muse")
        self.__midi_obj.addTempo(self.__track, time, 120)

    def add_sequence(self, seq):

        for t in range(seq.shape[0]):
            notes = MidiWriter.__get_note(seq[t])
            for note in notes:
                self.__midi_obj.addNote(self.__track, self.__channel, note, t, self.__duration, self.__volume)

    @staticmethod
    def __get_note(a):
        indexes = numpy.asarray(numpy.nonzero(a)) + 21
        return indexes[0]

    def write_to_file(self, filename):
        binfile = open(filename+".mid", 'wb')
        self.__midi_obj.writeFile(binfile)
        binfile.close()

if __name__ == '__main__':

    dataset = MuseDataset(seed=1344, pickle_file_path=Paths.muse_path, mode='full')
    seq = dataset.get_train_batch(1).inputs[:, :, 0]

    writer = MidiWriter()
    writer.add_sequence(seq)
    writer.write_to_file("out")
