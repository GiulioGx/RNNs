import abc

from training.Rule import Rule


class UpdateRule(Rule):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_update(self, net, lr, direction):
        """returns the update list with the new params for the theano function"""
