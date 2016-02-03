import abc

__author__ = 'giulio'


class SimpleInfoProducer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def infos(self):
        """return infos about itself"""

    def __str__(self):
        return str(self.infos)
