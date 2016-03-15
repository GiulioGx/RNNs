import abc

from infos.InfoProducer import SimpleInfoProducer
from task import Batch

__author__ = 'giulio'


class Task(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_batch(self, batch_size: int)->Batch:
        """return a batch of 'batch_size' training examples """
        return

    @abc.abstractmethod
    def error_fnc(self, t, y, mask):
        """return the 'true' error of output y wrt the labels t (theano symbols)"""
        return

    @abc.abstractproperty
    def n_in(self):
        """return the number of input units used for this task"""

    @abc.abstractproperty
    def n_out(self):
        """return the number of ouyput units used for this task"""
