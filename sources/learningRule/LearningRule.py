import abc

from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.SimpleInfoProducer import SimpleInfoProducer
from infos.SymbolicInfoProducer import SymbolicInfoProducer

__author__ = 'giulio'


class LearningStepRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, net, obj_fnc, dir_symbols: DescentDirectionRule.Symbols):
        """return the compiled version"""

    class Symbols(SymbolicInfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractmethod
        def learning_rate(self):
            """return a theano expression for the learning rate """



