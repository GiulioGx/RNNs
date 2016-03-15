import pickle

import numpy

from Configs import Configs
from infos.InfoElement import SimpleDescription
from datasets.Batch import Batch
from datasets.Dataset import Dataset


class MuseDataset(Dataset):
    def __init__(self, pickle_file_path: str, seed: int = Configs.seed, mode: str = 'split'):  # TODO enum
        dataset = pickle.load(open(pickle_file_path, "rb"))
        self.__min_note_index = 21  # inclusive
        self.__max_note_index = 108  # inclusive
        self.__n_notes = self.__max_note_index - self.__min_note_index
        self.__rng = numpy.random.RandomState(seed)
        self.__train_set = self.__pre_process_sequences(dataset['train'])
        self.__validation_set = self.__pre_process_sequences(dataset['valid'])
        self.__test_set = self.__pre_process_sequences(dataset['test'])

        if mode == 'split':
            self.__build_batch = self.__build_batch_splitted
        elif mode == 'full':
            self.__build_batch = self.__build_batch_full
        else:
            raise AttributeError('unsupported mode: {}. Available mode are "split" or "full" ')

    def get_train_batch(self, batch_size: int):
        indexes = self.__rng.randint(len(self.__train_set), size=(batch_size,))
        return self.__build_batch(self.__train_set, indexes)

    def __pre_process_sequences(self, sequences):
        max_length = 0
        processed_sequences = []
        for s in sequences:
            length = len(s)
            max_length = max(max_length, length)
            inputs = numpy.zeros((length, self.n_in), dtype=Configs.floatType)
            for t in range(length):
                indexes = numpy.asarray(s[t], dtype='int32') - self.__min_note_index
                inputs[t, indexes] = 1  # played note
            processed_sequences.append(inputs)
        print('max_length: {}'.format(max_length))  # TODO remove print
        return processed_sequences

    @staticmethod
    def __max_length(examples, indexes):
        return examples[max(indexes, key=lambda i: examples[i].shape[0])].shape[0]

    @staticmethod
    def __min_length(examples, indexes):
        return examples[min(indexes, key=lambda i: examples[i].shape[0])].shape[0]

    def __init_batch(self, length, batch_size):
        inputs = numpy.zeros(shape=(length, self.n_in, batch_size), dtype=Configs.floatType)
        outputs = numpy.zeros_like(inputs, dtype=Configs.floatType)
        mask = numpy.zeros_like(inputs, dtype=Configs.floatType)
        return inputs, outputs, mask

    def __build_batch_splitted(self, examples, indexes):
        fixed_length = 300  # FIXME magic constant
        bacth_size = len(indexes)
        min_length = MuseDataset.__min_length(examples, indexes) - 1
        batch_length = min(min_length, fixed_length)
        # print('bl', batch_length)
        inputs, outputs, mask = self.__init_batch(batch_length, bacth_size)

        for i in range(len(indexes)):
            example = examples[indexes[i]]
            length = len(example)
            # print('l', length)
            start = self.__rng.randint(low=0, high=length - batch_length)
            # print('start', start)
            inputs[0:batch_length, :, i] = example[start:start + batch_length, :]
            outputs[0:batch_length, :, i] = example[start + 1:start + batch_length + 1, :]
            mask[0:batch_length, :, i] = 1
        return Batch(inputs, outputs, mask)

    def __build_batch_full(self, examples, indexes):

        max_length = MuseDataset.__max_length(examples, indexes)
        bacth_size = len(indexes)
        inputs, outputs, mask = self.__init_batch(max_length - 1, bacth_size)

        for i in range(len(indexes)):
            example = examples[indexes[i]]
            length = len(example)
            inputs[0:length - 1, :, i] = example[0:length - 1, :]
            outputs[0:length - 1, :, i] = example[1:length, :]
            mask[0:length - 1, :, i] = 1
        return Batch(inputs, outputs, mask)

    @property
    def n_in(self):
        return self.__n_notes

    @property
    def n_out(self):
        return self.__n_notes

    @property
    def infos(self):

        description = ['Muse Dataset:\n',
                       '{} train sequences \n'.format(len(self.__train_set)),
                       '{} test sequences \n'.format(len(self.__test_set)),
                       '{} validation sequences \n'.format(len(self.__validation_set))]

        return SimpleDescription(''.join(description))

    def __process_set(self, set):
        batches = []
        for i in range(len(set)):
            batches.append(self.__build_batch_full(set, [i]))
        return batches

    @property
    def validation_set(self):
        return self.__process_set(self.__validation_set)

    @property
    def test_set(self):
        return self.__process_set(self.__test_set)

    @property
    def train_set(self):
        return self.__process_set(self.__train_set)


if __name__ == '__main__':
    numpy.set_printoptions(threshold=numpy.inf)
    seed = 545
    data_path = '/home/giulio/RNNs/datasets/polyphonic/musedata/MuseData.pickle'
    dataset = MuseDataset(seed=seed, pickle_file_path=data_path)
    print(str(dataset.infos))
    batch = dataset.get_train_batch(1)
    print(str(batch))
    print('input batch shape: ', batch.inputs.shape)
    print('output batch shape: ', batch.outputs.shape)
