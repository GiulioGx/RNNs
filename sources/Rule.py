import abc

__author__ = 'giulio'


class Rule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def format_infos(self, infos):
        """return a string representation of the infos produced by the rule and the consumed infos list"""
        return
