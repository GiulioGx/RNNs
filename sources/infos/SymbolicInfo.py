import abc

from infos.Info import Info, NullInfo

__author__ = 'giulio'


class SymbolicInfo(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fill_symbols(self, symbols_replacedments: list) -> Info:
        """instanziate a 'Info' object given a list of symbol replacement (i.e. theano functions outputs)"""

    @abc.abstractproperty
    def symbols(self):
        """return a list of symbols (theano outputs) to returned by a theano function"""


class NullSymbolicInfos(SymbolicInfo):
    @property
    def symbols(self):
        return []

    def fill_symbols(self, symbols_replacedments: list) -> Info:
        return NullInfo()
