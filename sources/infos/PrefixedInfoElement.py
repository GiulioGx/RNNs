from infos.InfoElement import InfoElement

__author__ = 'giulio'


class PrefixedInfoElement(InfoElement):

    def __init__(self, e: InfoElement, prefix: str):
        self.__prefix = prefix
        self.__element = e

    @property
    def value(self):
        return self.__element.value

    @property
    def name(self):
        return self.__prefix + '_' + self.__element.name

    @property
    def elements(self):
        return [self]

    @property
    def descr(self):
        return self.__element.descr
