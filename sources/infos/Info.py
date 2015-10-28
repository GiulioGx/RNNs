import abc

__author__ = 'giulio'


class Info(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def descr(self):
        """description"""

    @abc.abstractproperty
    def elements(self):
        """return the list of InfoElements the object is composed of"""

    @property
    def length(self):
        return len(self.elements)

    def __str__(self):
        return self.descr

    @property
    def dictionary(self):
        dictionary = {}
        for e in self.elements:
            if e.name not in dictionary:
                dictionary[e.name] = e.value
        return dictionary


class NullInfo(Info):

    @property
    def descr(self):
        return ''

    @property
    def elements(self):
        return []











