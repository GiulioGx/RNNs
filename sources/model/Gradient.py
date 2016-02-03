import abc

from combiningRule.CombiningRule import CombiningRule
from oldies.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class Gradient(SymbolicInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def temporal_combination(self, strategy: CombiningRule):
        """return a 'Combination' class which represents the combine of the gradients defined by 'strategy'"""

    @abc.abstractproperty
    def value(self):
        """returns the gradients represented by class Variables"""

    @abc.abstractproperty
    def loss_value(self):
        """return the value of the loss function"""


