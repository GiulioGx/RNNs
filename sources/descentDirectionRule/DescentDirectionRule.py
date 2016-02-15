import abc

from ObjectiveFunction import ObjectiveFunction
from training.Rule import Rule

__author__ = 'giulio'


class DescentDirectionRule(Rule):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def direction(self, net, obj_fnc:ObjectiveFunction):
        """returns a direction"""

