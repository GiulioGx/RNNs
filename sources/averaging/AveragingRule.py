import abc
from infos.SimpleInfoProducer import SimpleInfoProducer
from infos.SymbolicInfoProducer import SymbolicInfoProducer
from model import Variables


class AveragingRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compile(self, net, update_params: Variables):
        """returns the compiled Symbols"""

    class Symbols(SymbolicInfoProducer):
        __metaclass__ = abc.ABCMeta

        @abc.abstractproperty
        def averaged_params(self):
            """return the new params according to the average rule"""

        @abc.abstractproperty
        def update_list(self):
            """list of theano updates"""
