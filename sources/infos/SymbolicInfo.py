import abc

from infos.Info import Info

__author__ = 'giulio'


class SymbolicInfo(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fill_symbols(self, symbols_replacedments:list)->Info:
        """instanziate a 'Info' object given a list of symbol replacement (i.e. theano functions outputs)"""
