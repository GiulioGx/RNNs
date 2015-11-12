import abc
from infos.Info import Info

__author__ = 'giulio'


class InfoElement(Info):
    __metaclass__ = abc.ABCMeta

    @property
    def elements(self):
        return [self]

    @abc.abstractproperty
    def name(self):
        """name"""

    @abc.abstractproperty
    def value(self):
        """value"""

class SimpleDescription(InfoElement):

    def __init__(self, descr):
        self.__descr = descr

    @property
    def value(self):
        return self.__descr

    @property
    def name(self):
        return 'description'

    @property
    def descr(self):
        return self.__descr


class NonPrintableInfoElement(InfoElement):

    def __init__(self, name: str, value):
        self.__name = name
        self.__value = value

    @property
    def name(self):
        return self.__name

    @property
    def value(self):
        return self.__value

    @property
    def descr(self):
        return ''


class PrintableInfoElement(InfoElement):

    def __init__(self, name: str, style: str, value):
        self.__name = name
        self.__style = style
        self.__value = value

    @property
    def name(self):
        return self.__name

    @property
    def value(self):
        return self.__value

    @property
    def descr(self):
        s = self.__name+': {'+self.__style+'}'
        return s.format(self.__value)
