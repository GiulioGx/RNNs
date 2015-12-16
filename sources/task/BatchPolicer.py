import abc
from infos.InfoElement import SimpleDescription
from infos.SimpleInfoProducer import SimpleInfoProducer
from task.Dataset import Dataset

__author__ = 'giulio'


class BatchPolicer(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_train_batch(self):
        """return a batch according to some policy"""


class SimplePolicer(BatchPolicer):
    def __init__(self, dataset: Dataset, batch_size: int):
        self.__batch_size = batch_size
        self.__dataset = dataset

    def infos(self):
        return SimpleDescription('simple batch policer')

    def get_train_batch(self):
        return self.__dataset.get_train_batch(self.__batch_size)


class RepetitaPolicer(BatchPolicer):
    def __init__(self, dataset: Dataset, batch_size: int, n_repetitions=10, block_size=1000):
        self.__batch_size = batch_size
        self.__dataset = dataset
        self.__n_repetitions = n_repetitions
        self.__block_count = 0
        self.__repetition_count = 0
        self.__saved_batches = [None] * block_size
        self.__block_size = block_size

    def infos(self):
        return SimpleDescription('repetita batch policer')

    def get_train_batch(self):
        if self.__repetition_count == 0:
            self.__saved_batches[self.__block_count] = self.__dataset.get_train_batch(self.__batch_size)

        curr_batch = self.__saved_batches[self.__block_count]
        self.__block_count += 1

        if self.__block_count == self.__block_size:
            self.__block_count = 0
            self.__repetition_count = (self.__repetition_count + 1) % self.__n_repetitions

        return curr_batch
