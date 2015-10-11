import abc

__author__ = 'giulio'


class Task(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_batch(self, batch_size: int):
        """return a batch of 'batch_size' training examples """
        return

    @abc.abstractmethod
    def error_fnc(self, t, y):
        """return the 'true' error of output y wrt the labels t (theano symbols)"""
        return
