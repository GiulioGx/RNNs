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
    def __init__(self, dataset: Dataset, batch_size: int, n_repetition=10):
        self.__batch_size = batch_size
        self.__dataset = dataset
        self.__n_repetition = n_repetition
        self.__count = 0
        self.__current_batch = self.__dataset.get_train_batch(self.__batch_size)

    def infos(self):
        return SimpleDescription('simple batch policer')

    def get_train_batch(self):
        if self.__count == self.__n_repetition:
            self.__current_batch = self.__dataset.get_train_batch(self.__batch_size)
            self.__count = 1
        else:
            self.__count += 1
        return self.__current_batch
