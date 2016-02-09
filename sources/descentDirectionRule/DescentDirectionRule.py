import abc

from ObjectiveFunction import ObjectiveFunction
from Rule import Rule
from infos.InfoProducer import SimpleInfoProducer
from oldies.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class DescentDirectionRule(Rule):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def direction(self, net, obj_fnc:ObjectiveFunction):
        """returns a direction"""

