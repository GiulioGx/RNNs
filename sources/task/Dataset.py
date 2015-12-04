import abc

from task import Task
from task.Batch import Batch
import numpy


class Dataset(object):  # TODO change name
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def validation_set(self):
        """returns the validation 'Batch'"""

    @abc.abstractmethod
    def get_train_batch(self, batch_size: int):
        """returns a training batch"""

    @abc.abstractmethod
    def computer_error(self, t, y):
        """return the 'true' error of output y wrt the labels t (theano symbols)"""

    def no_valid_dataset_from_task(self, task: Task, size: int):
        examples = task.get_batch(size)
        return FiniteDataset(examples, examples, task.error_fnc)


class InfiniteDataset(Dataset):

    def __init__(self, task: Task, validation_size: int):
        self.__validation_size = validation_size
        self.__task = task

    def get_train_batch(self, batch_size: int):
        return self.__task.get_batch(batch_size)

    def validation_set(self):
        return self.__task.get_batch(self.__validation_size)

    def computer_error(self, t, y):
        return self.__task.error_fnc(t, y)


class FiniteDataset(Dataset):

    def __init__(self, validation_set: Batch, training_set: Batch, error_fnc):  # TODO error_fnc class
        self.__validation_set = validation_set
        self.__training_set = training_set
        self.__n_examples = self.__training_set.inputs.shape[2]
        self.__error_fnc = error_fnc

    def get_train_batch(self, batch_size: int):

        if batch_size == self.__n_examples:
            return self.__training_set
        else:
            if batch_size < self.__n_examples:
                indexes = numpy.random.choice(self.__n_examples, batch_size, replace=False)
            elif batch_size > self.__n_examples:
                indexes = numpy.random.choice(self.__n_examples, batch_size, replace=True)
            return self.__training_set[:, :, indexes]

    @property
    def validation_set(self):
        return self.__validation_set

    def computer_error(self, t, y):
        return self.__error_fnc(t, y)
