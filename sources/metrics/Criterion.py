import abc

from infos.InfoElement import SimpleDescription
from infos.InfoProducer import SimpleInfoProducer

__author__ = 'giulio'


class Criterion(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def is_satisfied(self) -> bool:
        """return  'True' or 'False' wheter ther criterion is satisifed or not"""


class AlwaysTrueCriterion(Criterion):

    def is_satisfied(self) -> bool:
        return True

    @property
    def infos(self):
        return SimpleDescription("Always True criterion")


class AlwaysFalseCriterion(Criterion):

    def is_satisfied(self) -> bool:
        return False

    @property
    def infos(self):
        return SimpleDescription("Always False criterion")