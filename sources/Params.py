import abc

__author__ = 'giulio'


class Params(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __add__(self, other):
        """addition with params operation"""

    @abc.abstractmethod
    def __mul__(self, alpha):
        """scalar multiplication"""

    @abc.abstractmethod
    def norm(self):
        """euclidean norm"""

    @abc.abstractmethod
    def grad(self, fnc):
        """return the gradient of the function fnc wrt the parameters this class represents"""

    @abc.abstractmethod
    def update_dictionary(self, other):
        """return the update dictionary for a theano function"""

