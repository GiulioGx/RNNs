import abc

from infos.InfoProducer import SimpleInfoProducer

__author__ = 'giulio'


class Criterion(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_satisfied(self) -> bool:
        """return  'True' or 'False' wheter ther criterion is satisifed or not"""
