import abc

from infos.Info import NullInfo
from infos.InfoElement import PrintableInfoElement
from infos.InfoGroup import InfoGroup
from infos.InfoList import InfoList
from infos.InfoProducer import SimpleInfoProducer
from task import Task
from task.Batch import Batch
import numpy


class Dataset(SimpleInfoProducer):  # TODO change name
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def n_in(self):
        """"""

    @abc.abstractproperty
    def n_out(self):
        """"""

    @abc.abstractproperty
    def validation_set(self):
        """returns a list of validation 'Batch'es"""

    @abc.abstractmethod
    def get_train_batch(self, batch_size: int):
        """returns a training batch"""

    @abc.abstractmethod
    def computer_error(self, t, y):
        """return the 'true' error of output y wrt the labels t (theano symbols)"""

    @staticmethod
    def no_valid_dataset_from_task(task: Task, size: int):
        examples = task.get_batch(size)
        infos = InfoGroup('finite_dataset_no_val',
                          InfoList(PrintableInfoElement('n_examples', ':02d', size), task.infos))
        return FiniteDataset(examples, examples, task.error_fnc, infos)


class InfiniteDataset(Dataset):
    def __init__(self, task: Task, validation_size: int):
        self.__validation_size = validation_size
        self.__task = task
        self.__infos = InfoGroup('infinite dataset', InfoList(PrintableInfoElement('validation_set_size', ':d',
                                                                                   validation_size), task.infos))

    def get_train_batch(self, batch_size: int):
        return self.__task.get_batch(batch_size)

    @property
    def validation_set(self):
        return [self.__task.get_batch(self.__validation_size)]  # XXX FIXME add different lenghts

    def computer_error(self, t, y):
        return self.__task.error_fnc(t, y)

    @property
    def n_in(self):
        return self.__task.n_in

    @property
    def n_out(self):
        return self.__task.n_out

    @property
    def infos(self):
        return self.__infos


class FiniteDataset(Dataset):
    def __init__(self, validation_set: Batch, training_set: Batch, error_fnc, infos=NullInfo()):  # TODO error_fnc class
        self.__validation_set = validation_set
        self.__training_set = training_set
        self.__n_examples = self.__training_set.inputs.shape[2]
        self.__error_fnc = error_fnc
        self.__n_in = training_set.sequences_dims[0]
        self.__n_out = training_set.sequences_dims[1]
        self.__infos = infos

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

    @property
    def n_in(self):
        return self.__n_in

    @property
    def n_out(self):
        return self.__n_out

    @property
    def infos(self):
        return self.__infos
