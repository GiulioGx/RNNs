import abc

from descentDirectionRule.DescentDirectionRule import DescentDirectionRule
from infos.SimpleInfoProducer import SimpleInfoProducer
from infos.SymbolicInfoProducer import SymbolicInfoProducer
from learningRule.LearningRule import LearningStepRule


class UpdateRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, net, net_symbols, lr_symbols:LearningStepRule.Symbols, dir_symbols:DescentDirectionRule.Symbols):
        """returns the compiled Symbols"""

    class Symbols(SymbolicInfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractproperty
        def update_list(self):
            """list of theano updates"""
