import abc

from infos.InfoProducer import SimpleInfoProducer


class UpdateRule(SimpleInfoProducer):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_update(self, net, lr, direction):
        """returns the update list with the new params for the theano function"""
