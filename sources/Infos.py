import abc

__author__ = 'giulio'


class Info(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def descr(self):
        """description"""

    @abc.abstractmethod
    def elements(self):
        """return the list of InfoElements the object is composed of"""

    def __str__(self):
        return self.descr


class InfoElement(Info):
    __metaclass__ = abc.ABCMeta

    @property
    def elements(self):
        return [self]

    @abc.abstractmethod
    def name(self):
        """name"""

    @abc.abstractmethod
    def value(self):
        """value"""


class NullInfo(Info):

    @property
    def descr(self):
        return ''

    @property
    def elements(self):
        return []


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


class InfoList(Info):

    def __init__(self, *info_list: Info):
        self.__info_list = info_list

    @property
    def elements(self):
        l = []
        for e in self.__info_list:
            l += e.elements
        return l

    @property
    def descr(self):
        s = ''
        t = ''
        for e in self.__info_list:
            d = e.descr
            if d != '':
                s = s + t + e.descr
                t = ', '
        return s


class InfoGroup(Info):

    @property
    def elements(self):
        return self.__info_list.elements

    def __init__(self, category_descr, info_list: InfoList):
        self.__category_descr = category_descr
        self.__info_list = info_list

    @property
    def descr(self):
        s = self.__category_descr + '=[' + self.__info_list.descr+']'
        return s











