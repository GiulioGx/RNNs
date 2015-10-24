import abc

__author__ = 'giulio'


class InfoProducer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def format_infos(self, infos):
        """return a representation (Info) of the infos produced by the rule and the consumed infos list"""

    @abc.abstractmethod
    def infos(self):
        """return a list of symbols of informations of various kind"""
