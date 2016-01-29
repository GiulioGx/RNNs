import pickle

import numpy

from Configs import Configs
from infos.InfoElement import SimpleDescription
from task.Batch import Batch
from task.Dataset import Dataset


class MuseDataset(Dataset):
    def __init__(self, pickle_filep_path: str, seed: int = Configs.seed):
        dataset = pickle.load(open(pickle_filep_path, "rb"))
        self.__min_note_index = 21  # inclusive
        self.__max_note_index = 108  # inclusive
        self.__n_notes = self.__max_note_index - self.__min_note_index  # FOXME *2??
        self.__rng = numpy.random.RandomState(seed)
        self.__train = self.__pre_process_sequences(dataset['train'])
        self.__validation = self.__pre_process_sequences(dataset['valid'])

    def get_train_batch(self, batch_size: int):
        example_index = self.__rng.randint(len(self.__train))
        example = self.__train[example_index]
        return MuseDataset.__build_batch(example)

    def __pre_process_sequences(self, sequences):
        processed_sequences = []
        for s in sequences:
            length = len(s)
            inputs = numpy.zeros((length, self.n_in, 1), dtype=Configs.floatType)
            for t in range(length):
                indexes = numpy.asarray(s[t], dtype='int32')-self.__min_note_index
                inputs[t, indexes, 0] = 1  # played note
            processed_sequences.append(inputs)
        return processed_sequences

    @staticmethod
    def __build_batch(example):

        batch_size = 1  # FIXME bacth_size is ignored
        length = len(example)
        inputs = example[0:length-1, :, :]
        outputs = example[1:length, :, :]
        return Batch(inputs, outputs)

    @property
    def n_in(self):
        return self.__n_notes

    @property
    def computer_error(self, t, y):
        return 1  # XXX not sure what to do. is this meaningful for this dataset?

    @property
    def n_out(self):
        return self.__n_notes

    @property
    def infos(self):
        return SimpleDescription('MuseDataset')

    @property
    def validation_set(self):
        batches = []
        for example in self.__validation:
            batches.append(self.__build_batch(example))
        return batches


if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.inf)
    seed = 99
    pickle_file_path = '/home/giulio/RNNs/datasets/polyphonic/musedata/MuseData.pickle'
    print('Testing MuseDataset ...')
    task = MuseDataset(seed = seed, pickle_filep_path=pickle_file_path)
    batch = task.get_train_batch(1)
    print(str(batch))

