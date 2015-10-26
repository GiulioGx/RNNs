import abc

from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from InfoProducer import InfoProducer

__author__ = 'giulio'


class LearningStepRule(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, net, obj_fnc, dir_symbols: DescentDirectionRule.Symbols):
        """return the compiled version"""


class LearningStepSymbols(InfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def learning_rate(self):
        """return a theano expression for the learning rate """
