import abc

from Rule import Rule
from infos.InfoProducer import SimpleInfoProducer
from oldies.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class DescentDirectionRule(Rule):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def direction(self, net_symbols, obj_symbols):
        """returns a direction"""

